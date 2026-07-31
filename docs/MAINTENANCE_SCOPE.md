# Maintenance scope

GADplus contains a research archive accumulated through many exploratory
campaigns.  This document prevents historical files from silently competing
with the maintained implementation.

## Maintained code

| Area | Maintained files |
| --- | --- |
| GAD | `src/gadplus/search/gad_search.py`, `scripts/gad_smoke.py` |
| Sella | `scripts/sella_smoke.py`, `src/gadplus/calculator/sella.py` |
| g-xTB | `src/gadplus/calculator/gxtb.py`, `scripts/run_gxtb_*.slurm`, `scripts/trillium/run_gxtb_*.slurm` |
| IRC_TOPO | `scripts/gxtb_irc_topo.py`, `src/gadplus/search/irc_full_hessian.py`, `src/gadplus/search/native_endpoints.py` |
| projection/gates | `src/gadplus/projection/`, `src/gadplus/core/convergence.py` |
| portable configuration | `configs/`, `scripts/setup_env.sh` |

## Historical material

- `scripts/archive/` contains historical experiment launchers and analyses.
  It is retained for provenance, not as a development surface.
- `legacy/` is immutable provenance, not maintained execution code.
- `docs/research/` contains dated result narratives and drafts, not operating
  instructions.
- Alternative search modules are not removed merely because they are not the
  default; many remain referenced by historical scripts.  Delete a module
  only in the same change that removes every executable reference to it.

## Removal rule

It is safe to delete generated output, caches, temporary environments, and
untracked binaries.  A tracked source file is removable only when:

1. `rg` finds no maintained or historical executable import/reference; and
2. it is not cited as a reproducibility artifact in a retained report.

This prevents a cosmetic cleanup from destroying the ability to reproduce a
published or archived result.
