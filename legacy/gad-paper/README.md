# gad-paper

Self-contained repo for writing the GAD TS-finding paper. Nothing here depends
on the experiment repo. Data is **HIP / Transition1x test split, n=287**.

## Compile
```
pdflatex paper.tex   # run twice for cross-refs → paper.pdf (21 pp)
```

## Layout
| Path | What |
|---|---|
| `paper.tex` | **The manuscript source — edit this.** (was `BENCHMARK_REPORT_2026-05-16.tex`) |
| `paper.pdf` | Last compiled reference |
| `figures/` | The 15 figures `paper.tex` includes (PDF) |
| `WRITING_PLAN.md` | Section-by-section readiness + **paper-section mapping** — the writing plan |
| `data/` | `master_2026_05_11.csv` (4-axis) + `threshold_sweep_2026_05_16.csv` — every table/figure traces here |
| `scripts/` | `build_pdf_2026_05_16.py` (figure generator) + `plotting_style.py` — to regen/tweak figures |
| `reference/HYBRID_FINDINGS_CATALOG.md` | The receipts — provenance for every hybrid claim |

## THE headline (verify against `data/master_2026_05_11.csv`)
Best-GAD vs best-Sella, test-287. TS-conv = `n_neg==1 ∧ fmax<0.01`; TOPO = fwd+bwd IRC graph-match.

| noise pm | TS-conv Δ (GAD−Sella) | IRC-TOPO Δ |
|---|---|---|
| 10 | −7.3 | −0.3 |
| 30 | −6.6 | +0.0 |
| 50 | −6.3 | +1.4 |
| 100 | +0.0 | +5.9 |
| 150 | +4.2 | +11.8 |
| 200 | **+17.4** | **+21.3** |

**Crossover, GAD wins BOTH metrics at high noise.** Sella for near-TS starts (≤50 pm);
GAD for far starts (≥150 pm) — at 200 pm **+17.4 pp TS-conv AND +21.3 pp TOPO**.
The hybrid's wall-time/converged-TS (1.4–3× vs Sella) is the separate "and it's cheap" prong.

## Gaps to fill (this is the work — no compute needed)
1. **Manuscript prose**: abstract, intro, related work (incl. noisyTS/LMHE Fig-3, 0–15 pm), methods prose, discussion, conclusion. Figures/tables already exist — see `WRITING_PLAN.md`.
2. **Headline prose**: the TeX TL;DR underplays the high-noise conv-rate win — fold in the crossover above.
3. Optional hygiene: drop the orphan `runs/starting_geom_300` line in §Sources.

## Settled (do not re-open)
- **SCINE — dropped** (high conv, IRC/TOPO ~0 above 10 pm; chemically empty).
- **Coordinate step — resolved**: Cartesian step + mass-weighted Eckart *projection* is canonical; genuine MW was a bug. Effect on GAD ≤0.35 pp on test-287, so reported numbers stand.
- **Hybrid numbers — valid** (all nonzero; a broken step can't reach a saddle).

## Excluded on purpose
Raw per-sample parquets (`runs/`, many GB) are **not** here. Figures are pre-built;
to regenerate from raw runs, use `scripts/` against the experiment repo
(`GAD_plus`) on Narval.
