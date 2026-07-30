"""Tests for exact local bundles and optional W&B replay preparation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow.parquet as pq
import torch

from gadplus.calculator.lennard_jones import (
    lj_atomic_nums,
    make_lj_predict_fn,
    pentagonal_bipyramid_geometry,
)
from gadplus.logging.pointwise import IntrinsicTrajectoryRecorder
from gadplus.logging.wandb_export import (
    deterministic_run_id,
    enrich_rows,
    event_preserving_indices,
    export_bundle,
    kabsch_rmsd,
    load_bundle,
)
from gadplus.projection import atomic_nums_to_symbols, get_mass_weights, vib_eig
from gadplus.search.intrinsic_gad import IntrinsicGADConfig, run_intrinsic_gad


class LocalBundleTests(unittest.TestCase):
    def test_observer_does_not_change_intrinsic_gad_coordinates(self) -> None:
        predictor = make_lj_predict_fn()
        atomic_numbers = lj_atomic_nums(7)
        minimum = pentagonal_bipyramid_geometry()
        symbols = atomic_nums_to_symbols(atomic_numbers)
        minimum_out = predictor(minimum, atomic_numbers, do_hessian=True)
        _, modes_mw, _ = vib_eig(minimum_out["hessian"], minimum, symbols)
        _, _, _, inv_sqrt_mass = get_mass_weights(symbols)
        start = minimum + 0.26 * (inv_sqrt_mass * modes_mw[:, 0]).reshape_as(minimum)
        config = IntrinsicGADConfig(
            max_steps=30,
            gate_variant="competitive",
            record_history=False,
        )

        baseline = run_intrinsic_gad(predictor, start, atomic_numbers, config)
        with tempfile.TemporaryDirectory() as temporary:
            recorder = IntrinsicTrajectoryRecorder(
                temporary,
                "lj7-observer-equivalence",
                atomic_numbers,
                config={**asdict(config), "sample_id": 7, "seed": 0},
            )

            def mutating_observer(observation) -> None:
                recorder(observation)
                # The optimizer must remain insulated even from a badly
                # behaved observer mutating the tensors it receives.
                observation.coords.zero_()
                observation.forces.zero_()
                observation.eigenvalues.zero_()

            observed = run_intrinsic_gad(
                predictor,
                start,
                atomic_numbers,
                config,
                observer=mutating_observer,
            )
            bundle = recorder.flush(observed, summary={"native_topology": True})

            torch.testing.assert_close(
                observed.final_coords,
                baseline.final_coords,
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(observed.converged, baseline.converged)
            self.assertEqual(len(recorder.rows), observed.n_evaluations)
            self.assertTrue(recorder.rows[-1]["terminal"])
            self.assertTrue(all(not row["terminal"] for row in recorder.rows[:-1]))
            self.assertIn("activity_fraction", recorder.rows[0])
            self.assertIn("lowest_reflection", recorder.rows[0])

            table = pq.read_table(bundle / "trajectory.parquet")
            self.assertEqual(table.num_rows, observed.n_evaluations)
            rows, coordinates, metadata = load_bundle(bundle)
            self.assertEqual(len(rows), len(coordinates))
            self.assertEqual(metadata["summary"]["native_topology"], True)
            self.assertEqual(metadata["schema_version"], 1)
            with (Path(bundle) / "metadata.json").open(encoding="utf-8") as handle:
                json.load(handle)

            with mock.patch.dict(
                os.environ,
                {
                    "WANDB_DIR": str(Path(temporary) / "wandb"),
                    "WANDB_SILENT": "true",
                },
            ):
                exported_id = export_bundle(
                    bundle,
                    mode="offline",
                    max_view_rows=25,
                    cockpit_chart_id="test-entity/gadplus-cockpit-v1",
                    mechanism_chart_id="test-entity/gadplus-mechanism-v1",
                )
            self.assertEqual(len(exported_id), 20)

    def test_kabsch_and_hindsight_enrichment(self) -> None:
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],
            dtype=np.float64,
        )
        angle = 0.6
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed = reference @ rotation + np.asarray([4.0, -3.0, 2.0])
        self.assertLess(kabsch_rmsd(transformed, reference), 1.0e-12)

        coordinates = np.stack([2.0 * reference, 1.5 * reference, reference])
        rows = [
            {"evaluation": index, "energy": float(index), "force_max": 0.1 / (index + 1)}
            for index in range(3)
        ]
        enriched = enrich_rows(
            rows,
            coordinates,
            force_threshold=0.01,
            labelled_ts=reference,
        )
        self.assertEqual(enriched[0]["force_ratio"], 10.0)
        self.assertAlmostEqual(enriched[-1]["terminal_progress_raw"], 1.0)
        self.assertAlmostEqual(enriched[-1]["distance_to_labelled_ts"], 0.0)


class CompactViewTests(unittest.TestCase):
    def test_event_preserving_view_is_bounded_and_keeps_crossings(self) -> None:
        rows = []
        for index in range(10_000):
            rows.append(
                {
                    "evaluation": index,
                    "n_neg": 2 if index < 4321 else 1,
                    "lambda2": -1.0 if index < 4321 else 1.0,
                    "force_max": 10.0 / (index + 1),
                    "effective_gate": 0.0 if index < 4321 else 1.0,
                    "step_cart_rms": 0.01,
                    "terminal": index == 9999,
                }
            )
        selected = event_preserving_indices(rows, max_rows=500)
        self.assertEqual(len(selected), 500)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 9999)
        self.assertIn(4320, selected)
        self.assertIn(4321, selected)
        self.assertEqual(selected, sorted(set(selected)))

    def test_deterministic_id_uses_full_identity(self) -> None:
        first = deterministic_run_id(["campaign", 17, 0.2, 0, "competitive-gad"])
        second = deterministic_run_id(["campaign", 17, 0.2, 0, "competitive-gad"])
        changed = deterministic_run_id(["campaign", 17, 0.2, 1, "competitive-gad"])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
