# Reaction Labeling and Glycine Proton-Transfer Selection

This note documents how we labeled the non-training Transition1x reactions and
selected a representative reaction for two-dimensional Hessian visualization.

## Goal

We wanted a chemically interpretable reaction from Transition1x for visualizing
how physically meaningful and locally accurate predicted Hessians are. The
target was a reaction with simple collective variables (CVs), so that a
two-dimensional scan could be interpreted directly in terms of bond making and
bond breaking near the transition state.

## Labeling Strategy

We first labeled every reaction outside the training set. The local
Transition1x HDF5 file contains the top-level splits `train`, `test`, and `val`,
so we excluded `train` and scanned `test` plus `val`.

The labeling was done with:

```bash
cd /lustre/fsw/portfolios/nvr/users/anburger/GAD_plus && uv run python scripts/label_transition1x_reactions.py --output-dir runs/reaction_labels --stem non_train_reaction_labels
```

The script writes:

- `runs/reaction_labels/non_train_reaction_labels.parquet`
- `runs/reaction_labels/non_train_reaction_labels.csv`

For each sample, the script builds reactant and product bond graphs from the
Transition1x geometries using ASE covalent-radius neighbor cutoffs. It then
compares the graph edges to identify bonds formed and broken. The labels are
topology-first: named reaction families are best-effort automatic hints and
should be treated as aids for selection rather than curated chemical ground
truth.

## Recorded Fields

For each non-training reaction we recorded the dataset identity and graph
changes:

- `split`, `sample_id`, `formula`, `rxn`, `n_atoms`, `has_product`
- `n_components_reactant`, `n_components_product`
- `topology_class`
- `n_bonds_formed`, `n_bonds_broken`
- `formed_bonds`, `broken_bonds`
- grouped element-pair counts such as `formed_CO`, `broken_CN`, `formed_HO`
- `reaction_center_atoms`, `n_reaction_center_atoms`,
  `reaction_center_elements`
- `reactant_ring_count`, `product_ring_count`, `ring_formed`, `ring_broken`,
  `delta_ring_count`
- `reaction_family`, `reaction_family_confidence`

The topology classes were defined from graph connectivity and bond changes:
`association`, `dissociation`, `intramolecular_rearrangement`, `exchange`,
`fragmentation_plus_rearrangement`, and `unknown`.

## Non-Training Set Summary

The scan labeled 512 non-training reactions:

- `test`: 287 reactions
- `val`: 225 reactions

Topology counts were:

- `test`: 217 intramolecular rearrangements, 52
  fragmentation-plus-rearrangements, 18 dissociations
- `val`: 136 intramolecular rearrangements, 76
  fragmentation-plus-rearrangements, 13 dissociations

Automatic reaction-family counts showed several classes with natural
two-dimensional CVs:

- proton transfer: 26 reactions
- H shift: 89 reactions
- substitution-like one-bond exchange: 83 reactions
- addition/ring-forming cases: 62 reactions
- elimination/fragmentation cases: 223 reactions

## Candidate Reaction Types

We considered several classes for two-dimensional scans:

- Proton transfer or H shift: use donor-H and acceptor-H distances. These are
  the cleanest Hessian diagnostics because the unstable mode should point along
  the transferring proton coordinate.
- Substitution-like bond exchange: use the broken-bond distance and formed-bond
  distance. These are simple one-bond-broken/one-bond-formed rearrangements.
- Elimination or fragmentation: use two breaking distances, or one breaking
  distance plus one forming distance. These are useful but can be less local and
  more pathway-dependent.
- Ring-forming additions: use the two forming bond distances. These give a
  familiar two-bond 2D surface, but the examples in the non-training set looked
  like small intramolecular ring closures rather than a clean Diels-Alder-type
  benchmark.

We found no obvious Diels-Alder reaction in the automatic non-training scan.

## Selected Reaction

We selected glycine intramolecular proton transfer:

- split: `test`
- sample: `sample_id=5`
- reaction id: `rxn1961`
- formula: `C2H5NO2`
- reaction family: `proton_transfer`

The graph-diff label for this reaction is:

- broken bond: `N-H`, atoms `(4, 9)`
- formed bond: `O-H`, atoms `(3, 9)`

The corresponding two-dimensional CVs are:

- `q1 = d(N4, H9)`
- `q2 = d(O3, H9)`

This reaction was selected because it is in the non-training set, has a single
chemically transparent proton-transfer event, and gives two directly
interpretable bond-distance coordinates. Near the transition state, a physically
meaningful Hessian should identify an unstable mode dominated by motion of the
transferring proton between the donor and acceptor atoms. Plotting the surface
in the two underlying distances, rather than only their difference, gives a
simple visual check of whether the predicted curvature captures the local
saddle geometry.

## Literature Motivation

Glycine is the simplest amino acid, with formula `C2H5NO2` and neutral structure
`NH2-CH2-COOH`. The proton-transfer process interconverts neutral glycine and
zwitterionic glycine:

```text
NH2-CH2-COOH -> NH3+-CH2-COO-
```

Prior studies often use a one-dimensional proton-transfer coordinate such as
`R = d(N-H) - d(O-H)`. Our two-dimensional scan uses the two underlying
distances separately, `d(N-H)` and `d(O-H)`, as a natural extension for Hessian
visualization.

Relevant literature includes:

- Leung and Rempe, "Ab initio Molecular Dynamics Study of Glycine
  Intramolecular Proton Transfer in Water":
  <https://export.arxiv.org/pdf/cond-mat/0503301v1.pdf>
- Kassab et al., "Theoretical study of solvent effect on intramolecular proton
  transfer of glycine":
  <https://doi.org/10.1016/S0166-1280(00)00451-6>
- Fernandez-Ramos et al., "A direct-dynamics study of the zwitterion-to-neutral
  interconversion of glycine in aqueous solution":
  <https://doi.org/10.1063/1.1322084>
- Zhang et al., "Intramolecular and Water Mediated Tautomerism of Solvated
  Glycine":
  <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c00273>

## Short SI Wording

We selected the glycine intramolecular proton-transfer reaction as a
representative two-dimensional benchmark for assessing the physical meaning of
predicted Hessians. This reaction is present in the non-training Transition1x
split (`test`, `sample_id=5`, `rxn1961`, formula `C2H5NO2`) and involves a
simple, chemically interpretable proton transfer from nitrogen to oxygen: the
N-H bond is broken while the O-H bond is formed. We therefore used the two bond
distances, `d(N-H)` and `d(O-H)`, as collective variables for visualizing the
local potential-energy surface. This choice provides a direct test of whether
the Hessian identifies the chemically relevant unstable direction, since the
transition-state mode should be dominated by motion of the transferring proton
between the donor and acceptor atoms. Glycine proton transfer is also a
well-studied model system in the literature, making it a recognizable and
chemically meaningful case for qualitative Hessian validation.

