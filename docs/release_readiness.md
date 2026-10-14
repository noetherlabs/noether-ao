# Public-release readiness

This is the release-preparation record for source commit
`fa8a8b8cf7f5f4bf35ff4b002119e05984592a00`. The repository remains private.
The audit prepares an owner decision; it does not grant a license, publish a
package or change repository visibility. The machine-readable result is retained
in [`audit.json`](../results/verification/release-readiness-20261014/audit.json).

## Verdict

**Not ready for public release.** Source integrity, package construction and the
credential scan pass. Publication remains blocked by explicit licensing, privacy,
distribution and support decisions.

| Area | Result | Evidence |
| --- | --- | --- |
| Current remote | private | `noetherlabs/noether-ao`; local and remote commit matched before the audit |
| Full-history credentials | pass | gitleaks 8.30.1 scanned 38 commits and 2,394 Git objects; zero findings |
| Retained numerical archives | pass | 1,372 NPZ files, 10,937 readable arrays, no string/object arrays |
| Pinned environment | pass | `uv lock --check`; 370 local tests and [GitHub Correctness](https://github.com/noetherlabs/noether-ao/actions/runs/38509422393) pass |
| Wheel build | pass | two builds produced the same SHA256; isolated install exposes all ten CLI commands |
| Wheel metadata | partial | README included and pytest removed from runtime; project license remains absent |
| Dragrace patch notice | pass | retained MIT text is byte-identical to the checked upstream license |
| Standard package resolution | block | wheel metadata does not reproduce the Git/URL pins encoded in `[tool.uv.sources]` |
| Public support route | block | no monitored security-reporting contact has been declared |

## Required owner decisions

1. **Choose the Noether AO license.** There is no root `LICENSE`, no license grant
   in package metadata and no license file in the wheel. Until the owner selects a
   license, the source is visible only under ordinary copyright.
2. **Choose a publication-history strategy.** `AGENTS.md` and the former `outbox/`
   paths are absent from the current tree but remain in four historical commits
   each. A public release can preserve that history, publish a reviewed/squashed
   history, or start a separate public repository. Do not rewrite the private
   research history implicitly.
3. **Choose how to handle workstation paths.** Thirty-five retained result files
   contain `/Users/...` provenance paths. They expose no credential, but reveal a
   local username and checkout structure. Editing those files in place would break
   recorded hashes. Accept them explicitly or publish separately derived, labeled
   copies while retaining the originals privately.
4. **Define the distribution contract.** The repository lock resolves `abcdLux`
   and `prysm` from pinned Git commits and `PyPROPER3` from a pinned URL. Built wheel
   metadata exposes a registry version for `abcdLux` and unpinned optional
   `prysm`/`pyproper3` requirements. Decide whether release is repository-only or
   make published dependency metadata reproduce the tested sources.
5. **Name a monitored security route.** Add `SECURITY.md` only after a real contact
   or issue route is assigned. No address is invented by this audit.
6. **Review the three unsent Dragrace drafts.** They contain no detected secret or
   private correspondence, but publication still needs human review and does not
   imply upstream acceptance.

## Package inspection

The clean source produced `noether_optics-0.1.0-py3-none-any.whl` twice with SHA256
`c2bbd3b09a4df36469d63ce6836e9abab8b55278073b19efcf6d84df26263ddf`.
It contains 26 members: 22 Python modules plus distribution metadata. A fresh
no-dependency install renders CLI help with all ten commands. The wheel now embeds
the README and omits pytest from `Requires-Dist`.

The wheel intentionally contains code only. Configurations, protocols, retained
results and contribution material remain repository artifacts. That boundary fits
the documented clone-first workflow, but a future package-index release needs
project URLs, a license and dependency-source resolution before upload.

## Direct license inventory

The table uses pinned installed-distribution metadata and bundled license texts.
No dependency source is vendored into the wheel. Binary redistribution still needs
the notices shipped by NumPy, SciPy, Matplotlib and their bundled components.

| Role | Packages | Recorded terms |
| --- | --- | --- |
| Build | hatchling 1.27.0 | MIT |
| Runtime | JAX/JAXlib 0.10.2; Equinox 0.13.8; Optax 0.2.8 | Apache-2.0 |
| Runtime | dLux 0.15.1; Zodiax 0.5.0; abcdLux 0.0.2; psutil 7.2.2 | BSD-3-Clause |
| Runtime | NumPy 2.4.6 | BSD-3-Clause, 0BSD, MIT, Zlib and CC0-1.0 components |
| Runtime | SciPy 1.17.1 | BSD family plus bundled third-party notices |
| Runtime | Matplotlib 3.11.2 | Matplotlib license plus bundled font licenses |
| Runtime | PyYAML 6.0.3; HCIPy 0.7.1 | MIT |
| Development | pytest 9.1.1 | MIT |
| Optional Dragrace | lentil 0.8.9; POPPY 1.1.2; prysm 0.22; PyPROPER3 3.3.7; threadpoolctl 3.7.0 | BSD/MIT family; lentil and PyPROPER variants need final notice review |
| Retained upstream patch | Diffraction-Dragrace | MIT; exact notice retained under `contributions/licenses/` |

This inventory is scoped to direct dependencies. Generate a complete transitive
software-bill-of-materials and notice bundle for the chosen release format after
the distribution contract is fixed.

## Reproduction

The audit used read-only repository and package checks. Secret values are never
written to the retained report.

```sh
gitleaks git --no-banner --redact=100 --report-format json .
uv lock --check
uv lock --check --no-sources
uv build --wheel --no-cache
CI=true JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

The `--no-sources` check is expected to fail until the public dependency contract
is resolved. The full-history credential scan and NPZ schema inspection should
remain clean. Re-run the audit after every licensing, history or packaging decision
and before changing repository visibility.
