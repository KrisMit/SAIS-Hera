# Reference implementation

Place the original Python implementation of **Layers 1, 3, and 4** here
as `sais_layers_1_3_4.py`, extracted from the SAIS v1.0 codebase as run
during Expedition Olympus.

This is included as **design/heritage reference only**. The Hera flight
port is a separate bare-metal C implementation and is not part of this
public repository.

## Before committing — cleanup checklist

Copy the relevant layers out of the original `SPACE AI SYSTEM (SAIS).py`,
then remove or correct the following so the public reference matches
what the code actually does:

- [ ] **Remove hard-coded result strings.** The mission-summary printout
      contains fixed values (e.g. a "2.3 W" power figure, "<300 ms"
      latency, and accuracy percentages) that are formatting constants,
      not measured or computed outputs. Delete them or replace with
      values the code actually computes — do not present them as results.
- [ ] **Keep only Layers 1, 3, 4.** Exclude the ML-ensemble (Layer 2)
      and the LLM-fallback (Layer 5) code paths — they are out of scope
      for Hera and their presence here would contradict the
      "no ML / no LLM runtime" design.
- [ ] **Describe models accurately.** If any comment or docstring refers
      to "LSTM" or "Dynamic Bayesian Networks," align it with what is
      implemented (a Bayesian update over discrete candidate failure
      modes; ensemble members, where present, are Isolation Forest /
      Gradient Boosting — but those belong to Layer 2 and are excluded
      here).
- [ ] **Strip any tooling artifacts.** Remove stray citation markers
      (e.g. `[cite: …]`) or other non-code annotations that may have
      been introduced by document tooling.
- [ ] **No ESA-official content.** Do not paste any interface names,
      data-pool parameters, packet structures, or numeric limits taken
      from the ESA "For ESA Official Use Only" annexes.
