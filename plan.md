
We are building a generative model for transition states. We are using diffusion based samplers that only require the score (proportional to the force) but no ground truth structures as training signal. Instead of the force we use the gentlest ascent dynamics calculated from analytical Hessians as a pseudo force that points towards transition states.

## Story

### Background/context

- Simulation is useful because many chemical processes are too small, too fast, or too expensive to observe in full atomic detail. Simulation gives a computational microscope for asking what molecular structures are likely and how they interconvert.
- Molecules can be modelled as collections of atoms whose positions determine their energy. A molecular simulation follows how these atomic positions evolve over time under the forces induced by the energy.
- Many important molecular events are transitions between stable structures, such as a molecule adopting a new shape, chemical reactions, or a drug docking to a protein. 
	- The most interesting events are often rare transitions between long-lived molecular states
	- many scientific questions reduce to understanding which molecular transformations are possible, including which reactions occur, which conformations are accessible, and which pathways dominate.
- Transitions are difficult to observe because molecules usually spend most of their time near stable minima and only rarely cross the high-energy bottlenecks separating them. Some transitions are only observed once every few millions to billions of simulation steps.
- Transition rates are controlled by the energy barrier between the stable states. The structure at the top of this barrier is the transition state. Finding transition states therefore leads to identifying reactions, their rates, and their mechanism

#### Technical background

* A molecule can be represented as a set of atomic coordinates
    $
    x \in \mathbb{R}^{3N},
    $
    where $N$ is the number of atoms.
* Chemical reactions occur on a potential energy surface
    $
    E(x),
    $
    which assigns an energy to each molecular geometry.
    * Stable molecules are local minima of $E(x)$. Transition states are index-1 saddle points where the energy gradient is zero, but the Hessian has one negative eigenvalue.
    $
    \nabla E(x^\ddagger)=0,\qquad
    \lambda_1(\nabla^2 E(x^\ddagger))<0,\quad
    \lambda_i(\nabla^2 E(x^\ddagger))>0\ \text{for } i>1.
    $

    * A force is the negative energy gradient.
    $
    F(x)=-\nabla E(x).
    $
* At thermal equilibrium, the molecular probability density is approximately
    $
    p(x)\propto e^{-\beta E(x)}.
    $
    Therefore its score is
    $
    \nabla_x \log p(x) = -\beta \nabla E(x)=\beta F(x).
    $
    Thus, for molecular systems, a diffusion sampler can often use the force as a guidance signal for score-based modelling.

* The pseudo-score is built from gentlest ascent dynamics. Let
    $
    H(x)=\nabla^2 E(x)
    $
      be the Hessian, and let $v_1(x)$ be the eigenvector corresponding to the smallest Hessian eigenvalue. The GAD vector field is defined as
    $
    F_{\mathrm{GAD}}(x)
    F(x)-2\big(F(x)^\top v_1(x)\big)v_1(x).
    $
    This keeps downhill motion in stable directions but reverses motion along the lowest-curvature direction, so trajectories are attracted toward index-1 saddle points rather than minima.

Why is it important now?
- Transition-state search remains a bottleneck in automated reaction discovery because conventional TS optimization is expensive, requires prior knowledge, and can fail to converge.
- Generative models have been applied to TS prediction, but many methods require hard-to-get data of TS geometries and known reactant-product pairs at inference time.
- Diffusion samplers have achieved strong results for sampling low energy / minima structures for equilibrium properties, but not for rare events to describe kinetics.
- This creates an opening for label-free, force-guided diffusion samplers that use physical information directly rather than relying on curated TS datasets.

### Motivation/Problem

What is the problem we are trying to solve?
* We want to generate molecular geometries that are close to transition states without requiring ground-truth transition-state structures.
* The problem is
    $
    \text{sample } x \approx x^\ddagger
    \quad\text{such that}\quad
    \nabla E(x)\approx 0
    \quad\text{and}\quad
    \operatorname{index}(\nabla^2 E(x))=1.
    $
* Existing generative TS models often treat TS generation as supervised learning from known TS geometries. Our project instead treats TS generation as score-guided sampling on the energy landscape.

Why is this problem important?
* Ground-truth TS structures are expensive to compute.
* TS datasets are smaller and less diverse than equilibrium molecular datasets.
* A label-free method could reduce the dependence on curated reaction datasets and make TS discovery more applicable to new molecules, new reactions, and poorly sampled chemical regimes.
* A successful sampler could produce candidate TS geometries that can then be refined by standard saddle-point optimization.

### Goal

Goals:
- coverage: discover a wide range of transition states
- generalize across reactions and systems

Constraints:
- no labels (data of transition state geometries)
- access to the energy, gradient (forces), Hessian of the potential energy surface (in the form of a force-field or MLIP)
- no prior knowledge about reaction (like reactant or product, collective variable)

Non-goals:
- target specific types of transitions, like reactions. Transition states can correspond to reactions, conformer changes, non-covalent bonds like docking, and others.
- target specific transitions like for certain endpoints. Narrowing to specific solutions can often be reasonably achieved through additional filtering or conditioning techniques later



### Prior work

- Optimization methods require prior knowledge about endpoints (NEB, GSM) or reaction coordinate (metadynamics, umbrella sampling)
- Generative models require data and endpoints of reactions

### Challenges

  * Diffusion-based samplers were designed for equilibrium, not transitions
  * The target distribution is not the usual equilibrium distribution.
    * Equilibrium sampling favors minima. TS sampling must favor rare saddle regions with low probability under the equilibrium distribution $p(x)\propto e^{-\beta E(x)}$.
  * A TS has mixed stability. Ordinary forces point toward energy minima. A valid TS is stable in $3N-1$ directions and unstable in one reaction direction. The sampler must move uphill in one direction while moving downhill in all others.
  * The GAD vector field only converges locally near index-1 saddle basins. Far from a saddle, the lowest-curvature eigenvector may not correspond to the chemically correct reaction coordinate.
  * Boltzmann formulation helps exploration and training stability, but can hinder converging to TS. Higher temperature smoothing improves stability and diversity. Lower termparature improves convergence to stationary points. The design must balance exploration against convergence to valid TS.

### Key Idea
* Instead of relying on hard-to-get data or reaction-specific optimization, we formulate the challenge of finding transition states as a sampling problem that we can target with (diffusion-based) learned samplers.
* We replace the learned force or supervised denoising target with a gentlest-ascent pseudo-force. We then use this vector field as the score-like guidance in a diffusion sampler.
    $
    F_{\mathrm{GAD}}(x) =
    F(x)-2(F(x)^\top v_1(x))v_1(x).
    $


How does this solve the challenges?
* Diffusion / Langevin annealing and the Boltzmann smoothing add exploration, so the method can sample multiple candidate TS basins rather than return one saddle like local optimizations do. 
* Diffusion is learned, so the model can generalize (amortize) across transitions and molecules
* Using GAD vector field avoids the need for ground-truth TS structures as training labels.

### Design details

* As score based sampler we choose adjoint sampling / adjoint schroedinger bridge sampling (ASBS) or bridge matching sampler (BMS) https://arxiv.org/abs/2603.00530v1, which only require the score at the final diffusion-time step, not at intermediate noisy steps
* To obtain energies, forces, Hessians we use the Hessian Interatomic Potential (HIP) model

* Post-processing
* Generated geometries are refined with a standard local TS optimizer (we will use our GAD or eigenvector-following-Newton).
* A candidate is accepted only if
    $
    |\nabla E(x)|<\epsilon
    \quad\text{and}\quad
    \operatorname{index}(\nabla^2E(x))=1.
    $

### Evaluation

#### Experiment: TS validity

* This experiment checks whether the sampler generates actual transition-state candidates, not just low-energy structures.
* We measure TS validity as
    $
    R_{\mathrm{TS}}
    \frac{1}{M}
    \sum_{i=1}^{M}
    \mathbf{1}
    \left[
    |\nabla E(x_i)|<\epsilon
    \ \land
    \operatorname{index}(H(x_i))=1
    \right].
    $
* The figure or table should report validity rate for

    * ordinary force-guided diffusion,
    * pure stochastic diffusion,
    * deterministic GAD,
    * our GAD-guided diffusion.
* Our method should produce a higher fraction of index-1 saddle candidates because its drift field makes saddles locally attractive.
* The method achieves the goal if it substantially improves $R_{\mathrm{TS}}$ over force-only and unguided baselines.

#### Experiment: Refinement success

* This experiment tests whether generated samples are useful starting points for TS optimization.
* We measure refinement success as
    $
    R_{\mathrm{refine}}
    \frac{{\text{generated samples converging to valid TSs}}}{{\text{generated samples}}}.
    $
* The figure or table should compare convergence rates and number of optimizer iterations needed after initialization from each method.
* Good samples should require fewer refinement steps and should fail less often.
* The sampler is useful if it generates geometries inside the basin of attraction of true saddle points.

* Final evaluation conclusion

* We conclude that GAD-guided diffusion achieves the project goal if it

    * generates a higher fraction of valid index-1 saddle points than force-only diffusion,
    * produces candidates that refine reliably into TSs,
    * matches benchmark TS geometries and barriers when references are available,
    * discovers multiple TS basins when multiple pathways exist,
    * and does so without using ground-truth TS structures as training labels.




## Thoughts and notes

### Wording
reactions -> transitions in general

diffusion-based samplers, score-based samplers, learned samplers

### Related work

https://www.nature.com/articles/s41467-023-44629-6 "Diffusion-based generative AI for exploring transition states from 2D molecular graphs | Nature Communications"

https://web.math.princeton.edu/string/gad/ "GAD"

https://openreview.net/forum?id=7brF4sMQq3 "GAP: Guided Diffusion for A Priori Transition State Sampling | OpenReview"

https://pubmed.ncbi.nlm.nih.gov/40652515/ "Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models - PubMed"

## Optional future ideas


* Experiment, Geometric accuracy on benchmark reactions

* This experiment verifies that generated TSs match known reference TSs when reference structures are available for evaluation only.
* We measure heavy-atom RMSD after alignment as
    $
    \operatorname{RMSD}
    \sqrt{
    \frac{1}{N}
    \sum_{i=1}^{N}
    |x_i^{\mathrm{gen}}-x_i^{\mathrm{ref}}|^2
    }.
    $
* The figure or table should plot RMSD distributions across benchmark reactions.
* A successful method should concentrate mass at low RMSD while still allowing multiple conformers when multiple TS basins exist.
* The approach achieves accurate TS generation if low-RMSD samples are obtained without using TS labels during training.

* Experiment, Barrier-height accuracy

* This experiment tests whether the generated TSs are chemically meaningful.
* We measure barrier-height error using
    $
    \Delta E^\ddagger = E(x^\ddagger)-E(x_{\mathrm{reactant}}),
    $
    and
    $
    \operatorname{MAE}_{\Delta E}
    \frac{1}{K}
    \sum_{k=1}^{K}
    |\Delta E_k^{\ddagger,\mathrm{gen}}
    \Delta E_k^{\ddagger,\mathrm{ref}}|.
    $
* The figure or table should report barrier MAE and rank correlation with reference barriers.
* A geometry may have low RMSD but still give a poor barrier, so this experiment checks chemical fidelity.
* The sampler is chemically useful if it recovers barriers close to reference TS calculations.

* Experiment 5, Diversity of discovered pathways

* This experiment shows whether diffusion contributes more than deterministic GAD by finding multiple TS basins.
* We measure the number of distinct refined TS clusters using RMSD or internal-coordinate clustering.
* The figure or table should show clusters of generated and refined TS structures with their corresponding barrier heights.
* Deterministic GAD may collapse to one nearby saddle, while stochastic GAD-guided diffusion can recover multiple reaction channels.
* The method combines local saddle attraction with global exploration.

# Plan

1. Find a vector field that converges to transition states (GAD, hybrid gradient descent and GAD, hybrid with eigenvector-following Newton)
2. Train ASBS (will clone later) on that vector field
3. Optional: pretrain ASBS on reactant/product structures, then train ASBS on that vector field