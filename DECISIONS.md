# Decisions

- Build isolated sibling repositories inside the writable project workspace:
  noether-optics and Diffraction-Dragrace. Do not modify prior Noether work.
- Sole engineer, as requested. No subagents. Record assumptions instead of asking
  questions. The autonomous request supersedes interactive gstack review prompts.
- Civil astronomy only. This is a simulation and measurement project, not a claim
  of operational assurance or collaboration with any institution.
- Configuration values, mathematical constants, version identifiers and acceptance
  tolerances are inputs, not measured results. All reported performance values must
  be generated in this session and linked to saved raw data and provenance.
- Use Python 3.11 and a uv lock. Keep source checkouts for source/API inspection.
- A single-plane FFT with explicit dynamic OPD is the initial latency contract.
  It is not the fixed focal-grid MFT contract in Dragrace. Report them separately.
  Multiwavelength FFT planes have wavelength-dependent angular grids; do not sum
  them into a broadband detector without resampling.
- macOS has no Linux sched_setaffinity. Never substitute environment thread flags
  for affinity. Unsupported execution modes get explicit skip records.
- The gstack scripts at the legacy path are absent, but the installed review skill
  and checklist are readable. Use their engineering/test/review workflow without
  installing or publishing unrelated changes.
- Gate tolerances and experiment configurations must be written before runs.
  Failed gates remain failures; fixes may correct implementation, never move a gate.
