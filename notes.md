
### Force Convergence criteria
Both 0.05 and 0.001 are valid convergence criteria for us.

ASE’s optimizer interface sets run(fmax=0.05, ...)  eV/Ang per default, and Sella inherits ASE’s Optimizer class.
Sella's readme recommends dyn.run(fmax=1e-3)
The widely used Gaussian standard uses 2.3e-2 for the max force component

Gaussian uses four criteria:
Maximum force
$$
F_{\max}=\max_k |g_k|,
$$
RMS Force
$$
F_{\mathrm{RMS}}=\left(\frac{1}{N}\sum_k g_k^2\right)^{1/2},
$$
Maximum displacement / step component
$$
\Delta q_{\max}=\max_k |\Delta q_k|,
$$
RMS displacement
$$
\Delta q_{\mathrm{RMS}}=\left(\frac{1}{N}\sum_k \Delta q_k^2\right)^{1/2}.
$$
| Criterion            | Gaussian default threshold / a.u. |            Converted threshold |
| -------------------- | --------------------------------: | -----------------------------: |
| Maximum Force        |               (0.000450\ E_h/a_0) | (0.02314\ \mathrm{eV\ Å^{-1}}) |
| RMS Force            |               (0.000300\ E_h/a_0) | (0.01543\ \mathrm{eV\ Å^{-1}}) |
| Maximum Displacement |                   (0.001800\ a_0) |        (0.0009525\ \mathrm{Å}) |
| RMS Displacement     |                   (0.001200\ a_0) |        (0.0006350\ \mathrm{Å}) |

Using the conversion from Hartree and Bohr to eV and Angstrom
$
1\ E_h/a_0 = 51.4220675\ \mathrm{eV\ Å^{-1}},
$
And conversion from Bohr to Angstrom:
$
1\ a_0 = 0.5291772105\ \mathrm{Å}.
$

