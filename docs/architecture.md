# SAIS-Hera — Architecture Notes

This document describes the three-layer decision pipeline in more
detail. It intentionally stays at the algorithm/design level and does
not reproduce any flight-target interface details.

## Overview

SAIS-Hera runs a single **acquire → evaluate → diagnose → report**
cycle per invocation. Each stage is bounded and deterministic, with no
dynamic allocation and no persistent state required between cycles
beyond a small diagnostic history log.

```
   telemetry snapshot
          │
          ▼
 ┌─────────────────────┐
 │ Layer 1: Fuzzy      │   fast, always-on safety filter
 │ Safety Guardian     │   (Mamdani-style inference)
 └─────────┬───────────┘
           │  deviation flagged?
           │  ── no ──► log nominal status, idle
           │  ── yes ─►
           ▼
 ┌─────────────────────┐
 │ Layer 3: Bayesian   │   root-cause reasoning over a
 │ Diagnostic Expert   │   small set of candidate failures
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ Layer 4:            │   synthesize + format a single
 │ Orchestrator        │   human-readable recommendation
 └─────────┬───────────┘
           ▼
   structured report (status / event / diagnostic)
```

## Layer 1 — Fuzzy Logic Safety Guardian

A deterministic Mamdani-style fuzzy inference system over a set of
monitored telemetry parameters. Triangular membership functions map
each input onto qualitative bands (e.g. low / nominal / high), a small
fixed rule base combines them, and defuzzification yields a scalar
risk/deviation score per cycle.

Key properties:
- **Constant-time** evaluation — a fixed rule set, no search, no growth.
- **Always on** — runs every cycle, independent of higher layers.
- **Interpretable** — each activation traces to a specific rule.

## Layer 3 — Bayesian Diagnostic Expert

Triggered only when Layer 1 flags a deviation. Applies Bayes' rule
over a small, pre-defined set of candidate failure modes (for example:
sensor degradation, power anomaly, thermal drift). Prior probabilities
and per-mode likelihoods are stored as fixed tables; the update yields
a posterior distribution over the candidates, and the highest-posterior
mode (with its confidence) becomes the diagnosis.

Key properties:
- Inference over a **small discrete hypothesis space** — bounded cost.
- **No training at runtime** — priors/likelihoods are static tables.
- **Explainable** — the posterior and contributing evidence are legible.

## Layer 4 — Orchestrator

Arbitrates between the Layer 1 safety state and the Layer 3 diagnosis,
decides what (if anything) is worth reporting, and formats a single
structured output: the flagged parameter(s), a confidence level, and
the most likely cause. Routine cycles produce a compact status
summary; confirmed deviations produce an event notification; deviations
warranting detail produce a fuller diagnostic write-up.

## Memory & compute model

- **Static allocation only.** All rule tables, probability tables, and
  buffers are fixed-size and declared at compile time.
- **No persistent state** across an involuntary stop. Each window
  re-initializes from static tables and runs a fresh cycle — consistent
  with an environment that can be halted without a graceful shutdown.
- **Small footprint.** No model weights, no embeddings; the working set
  is rule/probability tables plus a telemetry buffer.

## Reference implementation

`reference/sais_layers_1_3_4.py` contains the original Python
implementation of these three layers as exercised on edge hardware
during Expedition Olympus. It is included as design/heritage reference;
the flight port is a separate bare-metal C implementation and is not
part of this public repository.
