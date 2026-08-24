#!/usr/bin/env python3
"""
SAIS-Hera — Reference implementation of Layers 1, 3, and 4.

This is the interpretable, deterministic subset of the SAIS v1.0
architecture, extracted for the Hera OSIP experiment: the fuzzy-logic
safety guardian, the Bayesian diagnostic expert, and the decision
orchestrator that combines them.

The original SAIS v1.0 also included an ML-ensemble layer (Layer 2) and
an LLM fallback (Layer 5). Both are intentionally OMITTED here: they
depend on a Python/ML runtime with no bare-metal equivalent and carry a
far larger footprint, and neither is required for the
deterministic-plus-probabilistic reasoning this experiment demonstrates.
See docs/architecture.md for the rationale.

This Python code is a design/heritage reference (as exercised on
Jetson Xavier NX edge hardware during AATC Expedition Olympus). The
flight port is a separate bare-metal C implementation and is not part
of this public repository.

Author: Kristina Mitrović
"""

import numpy as np

# Fuzzy logic — scikit-fuzzy
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:  # pragma: no cover
    raise SystemExit(
        "This reference requires scikit-fuzzy. Install with:\n"
        "    pip install scikit-fuzzy networkx"
    )


# ===========================================================================
# LAYER 1 — FUZZY LOGIC SAFETY GUARDIAN
# ===========================================================================

class FuzzyLogicLayer:
    """Deterministic Mamdani-style fuzzy inference over safety telemetry.

    Maps temperature, remaining power, radiation, and system health onto
    a workload-reduction / risk output using triangular membership
    functions and a small fixed rule base. Always-on: evaluated every
    cycle, independent of the higher layers.
    """

    def __init__(self):
        self.setup_fuzzy_system()

    def setup_fuzzy_system(self):
        self.temperature = ctrl.Antecedent(np.arange(0, 120, 1), 'temperature')
        self.temperature['normal'] = fuzz.trimf(self.temperature.universe, [0, 50, 70])
        self.temperature['elevated'] = fuzz.trimf(self.temperature.universe, [60, 85, 95])
        self.temperature['critical'] = fuzz.trimf(self.temperature.universe, [90, 110, 120])

        self.power_remaining = ctrl.Antecedent(np.arange(0, 101, 1), 'power_remaining')
        self.power_remaining['abundant'] = fuzz.trimf(self.power_remaining.universe, [0, 75, 100])
        self.power_remaining['adequate'] = fuzz.trimf(self.power_remaining.universe, [40, 60, 80])
        self.power_remaining['low'] = fuzz.trimf(self.power_remaining.universe, [20, 40, 60])
        self.power_remaining['critical'] = fuzz.trimf(self.power_remaining.universe, [0, 20, 40])

        self.radiation = ctrl.Antecedent(np.arange(0, 100, 1), 'radiation')
        self.radiation['low'] = fuzz.trimf(self.radiation.universe, [0, 0, 5])
        self.radiation['medium'] = fuzz.trimf(self.radiation.universe, [3, 10, 20])
        self.radiation['high'] = fuzz.trimf(self.radiation.universe, [15, 50, 100])

        self.health = ctrl.Antecedent(np.arange(0, 101, 1), 'system_health')
        self.health['good'] = fuzz.trimf(self.health.universe, [60, 80, 100])
        self.health['degraded'] = fuzz.trimf(self.health.universe, [30, 50, 70])
        self.health['critical'] = fuzz.trimf(self.health.universe, [0, 20, 40])

        self.workload_reduction = ctrl.Consequent(np.arange(0, 101, 1), 'workload_reduction')
        self.workload_reduction['none'] = fuzz.trimf(self.workload_reduction.universe, [0, 0, 10])
        self.workload_reduction['light'] = fuzz.trimf(self.workload_reduction.universe, [5, 25, 45])
        self.workload_reduction['moderate'] = fuzz.trimf(self.workload_reduction.universe, [40, 60, 80])
        self.workload_reduction['heavy'] = fuzz.trimf(self.workload_reduction.universe, [75, 90, 100])
        self.workload_reduction['minimum'] = fuzz.trimf(self.workload_reduction.universe, [90, 100, 100])

        self.rules = [
            ctrl.Rule(self.power_remaining['critical'], self.workload_reduction['minimum']),
            ctrl.Rule(self.temperature['critical'], self.workload_reduction['heavy']),
            ctrl.Rule(self.radiation['high'], self.workload_reduction['moderate']),
            ctrl.Rule(self.health['critical'], self.workload_reduction['heavy']),
            ctrl.Rule(self.temperature['elevated'] & self.power_remaining['low'],
                      self.workload_reduction['moderate']),
            ctrl.Rule(self.temperature['normal'] & self.power_remaining['abundant'] & self.health['good'],
                      self.workload_reduction['none']),
        ]

        self.system = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def compute(self, temperature, power_remaining, radiation, system_health):
        """Return a scalar workload-reduction / risk score in [0, 100]."""
        try:
            self.sim.input['temperature'] = np.clip(temperature, 0, 120)
            self.sim.input['power_remaining'] = np.clip(power_remaining, 0, 100)
            self.sim.input['radiation'] = np.clip(radiation, 0, 100)
            self.sim.input['system_health'] = np.clip(system_health, 0, 100)
            self.sim.compute()
            return float(np.clip(self.sim.output['workload_reduction'], 0, 100))
        except Exception:
            # Fail safe to a mid-range value rather than crashing the loop.
            return 50.0


# ===========================================================================
# LAYER 3 — BAYESIAN DIAGNOSTIC EXPERT
# ===========================================================================

class BayesianDiagnosticsLayer:
    """Probabilistic root-cause diagnosis over candidate failure modes.

    Applies Bayes' rule over a small, fixed set of candidate failures
    using static prior and likelihood tables. Triggered by Layer 1 when
    a deviation is flagged. No runtime training; the hypothesis space is
    small and discrete, so cost is bounded.
    """

    def __init__(self):
        self.priors = {
            'cooling_failure': 0.05,
            'power_system_failure': 0.03,
            'thermal_degradation': 0.10,
            'battery_aging': 0.15,
            'radiation_damage': 0.08,
        }

        self.likelihoods = {
            'cooling_failure': {'high_temp': 0.90, 'low_efficiency': 0.85, 'normal_power': 0.80},
            'power_system_failure': {'high_power_draw': 0.75, 'low_battery': 0.88, 'high_temp': 0.40},
            'thermal_degradation': {'high_temp': 0.80, 'high_stress': 0.85, 'low_cooling': 0.70},
            'battery_aging': {'low_battery': 0.80, 'high_power_draw': 0.65, 'degraded_voltage': 0.85},
            'radiation_damage': {'high_seu': 0.90, 'memory_errors': 0.88, 'model_drift': 0.70},
        }

    def diagnose(self, observations):
        """Return a normalized posterior distribution over failure modes.

        `observations` is a dict of {symptom_name: bool}. For each mode
        the prior is multiplied by the likelihood of each present symptom
        (or its complement if absent), then the posteriors are normalized.
        """
        posteriors = {}
        for failure_mode in self.priors:
            posterior = self.priors[failure_mode]
            for symptom, present in observations.items():
                if symptom in self.likelihoods[failure_mode]:
                    likelihood = self.likelihoods[failure_mode][symptom]
                    posterior *= likelihood if present else (1 - likelihood)
            posteriors[failure_mode] = posterior

        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        return posteriors


# ===========================================================================
# LAYER 4 — DECISION ORCHESTRATOR
# ===========================================================================

class DecisionOrchestrator:
    """Combines Layer 1 (fuzzy) and Layer 3 (Bayesian) into one output.

    Runs the always-on fuzzy safety pass; when it flags a deviation,
    derives symptom evidence from the telemetry and runs a Bayesian
    diagnosis; then synthesizes a single, human-readable recommendation
    and a legible reasoning trace.
    """

    #: Fuzzy score above which the situation is treated as a deviation.
    DEVIATION_THRESHOLD = 50.0

    def __init__(self):
        self.fuzzy_layer = FuzzyLogicLayer()
        self.bayesian_layer = BayesianDiagnosticsLayer()
        self.decision_history = []

    def make_decision(self, sensor_data):
        """Run one acquire -> evaluate -> diagnose -> report cycle."""

        # --- Layer 1: fuzzy safety pass (always on) ---
        fuzzy_score = self.fuzzy_layer.compute(
            temperature=sensor_data.get('temperature', 50),
            power_remaining=sensor_data.get('power_remaining', 75),
            radiation=sensor_data.get('seu_rate', 5),
            system_health=sensor_data.get('health_index', 80),
        )

        deviation_flagged = fuzzy_score > self.DEVIATION_THRESHOLD

        # --- Layer 3: Bayesian diagnosis (only when Layer 1 flags) ---
        diagnosis = {}
        if deviation_flagged:
            symptoms = self._symptoms_from_telemetry(sensor_data)
            diagnosis = self.bayesian_layer.diagnose(symptoms)

        top_diagnosis = (
            max(diagnosis.items(), key=lambda kv: kv[1]) if diagnosis else None
        )

        # --- Layer 4: synthesize a single recommendation ---
        decision = {
            'layer1_fuzzy': {
                'risk_score': fuzzy_score,
                'action': self._fuzzy_action_from_value(fuzzy_score),
                'deviation_flagged': deviation_flagged,
            },
            'layer3_bayesian': {
                'diagnosis': {k: float(v) for k, v in diagnosis.items()},
                'top_diagnosis': top_diagnosis,
                'confidence': float(max(diagnosis.values())) if diagnosis else 0.0,
            },
            'recommendation': self._synthesize(fuzzy_score, diagnosis),
            'reasoning': self._generate_reasoning(sensor_data, fuzzy_score, diagnosis),
        }

        self.decision_history.append(decision)
        return decision

    @staticmethod
    def _symptoms_from_telemetry(sensor_data):
        """Map raw telemetry onto the boolean symptoms Layer 3 expects."""
        return {
            'high_temp': sensor_data.get('temperature', 50) > 80,
            'low_efficiency': sensor_data.get('cooling_efficiency', 90) < 70,
            'high_power_draw': sensor_data.get('power_draw', 50) > 100,
            'low_battery': sensor_data.get('battery_remaining', 75) < 40,
            'high_seu': sensor_data.get('seu_rate', 5) > 15,
            'memory_errors': sensor_data.get('uncorrectable_errors', 0) > 5,
            'model_drift': sensor_data.get('model_accuracy_drift', 0) > 5,
            'degraded_voltage': sensor_data.get('voltage', 5.0) < 4.8,
        }

    @staticmethod
    def _fuzzy_action_from_value(value):
        if value < 10:
            return "NONE - normal operation"
        elif value < 35:
            return "LIGHT - monitor closely"
        elif value < 65:
            return "MODERATE - reduce non-essential load"
        elif value < 85:
            return "HEAVY - minimize load, prioritize safety"
        else:
            return "MINIMUM - essential systems only"

    def _synthesize(self, fuzzy_score, diagnosis):
        actions = [f"Recommended load reduction: {fuzzy_score:.0f}%"]
        if diagnosis:
            top, prob = max(diagnosis.items(), key=lambda kv: kv[1])
            if prob > 0.3:
                actions.append(f"Likely cause: {top.replace('_', ' ')} (P={prob:.2f})")
        return actions

    def _generate_reasoning(self, sensor_data, fuzzy_score, diagnosis):
        reasons = []
        temp = sensor_data.get('temperature', 50)
        if temp > 90:
            reasons.append(f"Temperature critical at {temp}\u00b0C")
        elif temp > 80:
            reasons.append(f"Temperature elevated at {temp}\u00b0C")

        power = sensor_data.get('power_remaining', 75)
        if power < 20:
            reasons.append(f"Power critical ({power}%)")
        elif power < 40:
            reasons.append(f"Power low ({power}%)")

        if diagnosis:
            top, prob = max(diagnosis.items(), key=lambda kv: kv[1])
            if prob > 0.4:
                reasons.append(
                    f"Bayesian diagnosis suggests {top.replace('_', ' ')} "
                    f"({prob * 100:.0f}% confidence)"
                )
        return reasons


# ===========================================================================
# Minimal demonstration
# ===========================================================================

def _demo():
    """Run a few illustrative telemetry snapshots through the pipeline."""
    orchestrator = DecisionOrchestrator()

    snapshots = [
        {'label': 'nominal',
         'temperature': 45, 'power_remaining': 85, 'seu_rate': 3, 'health_index': 90},
        {'label': 'thermal event',
         'temperature': 98, 'power_remaining': 70, 'seu_rate': 4, 'health_index': 65,
         'cooling_efficiency': 60, 'power_draw': 60},
        {'label': 'power event',
         'temperature': 55, 'power_remaining': 25, 'seu_rate': 5, 'health_index': 60,
         'battery_remaining': 30, 'voltage': 4.6, 'power_draw': 120},
    ]

    for snap in snapshots:
        label = snap.pop('label')
        result = orchestrator.make_decision(snap)
        print(f"\n=== {label} ===")
        print(f"  fuzzy risk score : {result['layer1_fuzzy']['risk_score']:.1f}")
        print(f"  action           : {result['layer1_fuzzy']['action']}")
        print(f"  deviation flagged: {result['layer1_fuzzy']['deviation_flagged']}")
        if result['layer3_bayesian']['top_diagnosis']:
            cause, prob = result['layer3_bayesian']['top_diagnosis']
            print(f"  top diagnosis    : {cause} (P={prob:.2f})")
        print(f"  recommendation   : {result['recommendation']}")


if __name__ == "__main__":
    _demo()
