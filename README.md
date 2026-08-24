# SAIS-Hera

**Neurosymbolic onboard decision-support experiment for ESA's Hera mission.**

SAIS-Hera is a Hera-specific port of the interpretable core of
[SAIS v1.0](https://github.com/KrisMit/SAIS-v1.0) — a hybrid
neuro-symbolic decision architecture designed, implemented, and
field-tested during **AATC Expedition Olympus**, an 8-day analog Mars
mission (May 2026), where the author served as Mission Commander
(EXP106CDR).

This repository accompanies an Idea submission to ESA's OSIP call
*"Autonomous Software Experiments on Hera."*

---

## What it does

During Hera's extended mission phase, a communication round-trip with
ground control can take up to ~40 minutes one way. Any onboard anomaly
must currently wait for ground diagnosis before action — the spacecraft
can *observe* a problem but cannot *reason* about it independently.

SAIS-Hera is a lightweight, fully deterministic decision-support layer
that detects and diagnoses anomalies **onboard, in seconds**, directly
from spacecraft telemetry — no ground round-trip, no imaging payload,
no machine-learning or LLM runtime.

## Architecture

SAIS-Hera ports **three of the five original SAIS layers** — the
interpretable, low-footprint subset best suited to a flight-representative
sandbox:

| Layer | Role |
|-------|------|
| **Layer 1 — Fuzzy Logic Safety Guardian** | Deterministic Mamdani-style fuzzy inference; an always-on safety filter that scores telemetry against safety-envelope thresholds. |
| **Layer 3 — Bayesian Diagnostic Expert** | Applies Bayes' rule over a small set of candidate failure modes to identify the most probable root cause when Layer 1 flags a deviation. |
| **Layer 4 — Orchestrator** | Synthesizes Layers 1 and 3 into a single, human-readable recommendation and routes it to the appropriate report channel. |

### Deliberately excluded

The original architecture's **Layer 2 (ML Ensemble)** and
**Layer 5 (LLM fallback)** are intentionally left out of this port.
Both depend on a Python/ML runtime with no bare-metal equivalent, and
both carry a far larger power and memory footprint. Excluding them is
what keeps SAIS-Hera fully deterministic, auditable, and light enough
for a constrained embedded target — and none of it is needed for the
deterministic-plus-probabilistic reasoning this experiment demonstrates.

## Why neurosymbolic, and why interpretable

Onboard reasoning tends to force a trade-off:

- **Rule-based logic** is transparent and easy to certify, but brittle
  against novel or ambiguous faults.
- **ML / deep-learning** is more adaptive, but opaque, hard to certify,
  and the most resource-hungry option.

SAIS-Hera keeps the transparent, certifiable half (fuzzy safety +
Bayesian diagnosis) and treats interpretability as a hard requirement:
every flagged anomaly and its suggested cause can be traced back to
explicit rules and probabilities. That auditability is the point — it
is what makes onboard autonomy trustworthy enough to fly.

## Design goals (target environment)

- Fully deterministic, static-allocation-only (no dynamic memory).
- Telemetry-driven and asynchronous; never blocks other processes.
- Small, bounded resource footprint (no model weights or embeddings).
- Output only through standard, size-limited reporting channels —
  the system is a monitoring/diagnostic layer, not a data-generating
  instrument.

> Note on interfaces: the flight-target integration details (specific
> platform APIs, data-pool parameters, and telemetry packet structures)
> are derived from ESA documentation issued for the OSIP call under
> *"For ESA Official Use Only"* terms and are **not** included in this
> public repository. They are available to ESA evaluators on request.

## Repository contents

```
SAIS-Hera/
├── README.md                     ← this file
├── LICENSE
├── .gitignore
├── reference/
│   └── sais_layers_1_3_4.py      ← original Python implementation of
│                                    Layers 1, 3, 4 (Expedition Olympus
│                                    heritage; edge/Jetson target)
└── docs/
    ├── architecture.md           ← layer-by-layer design notes
    └── maturity.md               ← test heritage & TRL discussion
```

## Status

Idea phase (OSIP). Proof-of-concept only: field-exercised on edge
hardware during an analog mission, with **no prior heritage on
LEON3-class flight hardware**. Not radiation-hardened or
thermal-vacuum tested. The Hera sandbox would be the first validation
of the ported logic under space-representative constraints.

## Author

**Kristina Mitrović** — software engineer (hybrid neuro-symbolic
architectures for constrained, high-stakes environments) and analog
astronaut mission commander. Sole author and architect of SAIS.

## Related

- Original architecture: [SAIS-v1.0](https://github.com/KrisMit/SAIS-v1.0)
- ESA OSIP: [ideas.esa.int](https://ideas.esa.int)
