# SAIS-Hera — Maturity & Test Heritage

## Where the system has run

The three layers ported here (Fuzzy Safety Guardian, Bayesian
Diagnostic Expert, Orchestrator) were exercised as part of SAIS's
integrated architecture on **NVIDIA Jetson Xavier NX** edge hardware
during **AATC Expedition Olympus** — an 8-day analog Mars mission with
an 11-person international crew (May 2026), commanded by the author.

During the mission the system ran against live crew and environmental
telemetry and produced logged fuzzy-logic and diagnostic outputs under
mission-realistic operational conditions.

## Honest scope of that heritage

This was a **proof-of-concept deployment**, and it is described as such:

- End-to-end procedural accuracy, latency under load, and diagnostic
  precision were **not formally measured** during the mission.
- The system has **no radiation-hardening** and **no thermal-vacuum
  qualification**.
- It has **no prior heritage on LEON3-class flight hardware**.

## What the Hera experiment would add

A Hera sandbox run would be the **first validation** of the ported
logic — rewritten in bare-metal C, without the ML-ensemble or LLM
layers — under space-representative processing constraints. The
maturity step is explicit:

> from *"exercised in an analog operational environment on commodity
> edge hardware"* → to *"exercised in a real deep-space sandbox on
> flight-representative hardware."*

This is framed as a maturity increment, not as a re-flight of
previously space-qualified software.
