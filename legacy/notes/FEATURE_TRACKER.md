# Non-Fundamental Features (Not in v1)

Features that may improve performance but are NOT part of the core
GAD algorithm. Each would need independent justification if added.

## Optimization Enhancements
- [ ] RFO (Rational Function Optimization) step instead of pure NR
- [ ] P-RFO (Partitioned RFO) for TS refinement
- [ ] Trust radius management (Schlegel-style or ARC)
- [ ] Polynomial line search (cubic interpolation)
- [ ] GDIIS acceleration
- [ ] Levenberg-Marquardt damping

## Integration Methods
- [ ] RK45 adaptive integrator (replaces Euler)
- [ ] Predictor-corrector schemes

## Advanced Saddle Methods
- [ ] HiSD (High-index Saddle Dynamics)
- [ ] k-HiSD (generalized reflections)
- [ ] iHiSD (interpolated gradient flow → HiSD)
- [ ] Dimer method
- [ ] Growing string method

## Escape / Rescue Heuristics
- [ ] v2 multi-mode escape perturbation (kick along v2)
- [ ] Oscillation detection + kick
- [ ] Blind-mode kick (gradient orthogonal to negative modes)
- [ ] Late escape (aggressive displacement after many stagnant steps)
- [ ] Mode smoothing (beta < 1)

## Alternative Calculators
- [ ] SCINE Sparrow (semiempirical DFTB0/PM6/AM1)
- [ ] MACE / ANI
- [ ] Delta-ML (ML + DFT correction)
- [ ] Learned LMHE (GotenNet-GA leftmost Hessian eigenvector predictor)

## Alternative Starting Geometries
- [ ] NEB (Nudged Elastic Band) interpolation
- [ ] QST (Quadratic Synchronous Transit)
- [ ] Growing String Method initial guess

## Data / Analysis
- [ ] Optuna HPO integration
- [ ] Automated hyperparameter sensitivity analysis
- [ ] Trajectory replay / visualization tools
