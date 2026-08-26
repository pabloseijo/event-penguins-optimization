"""Tests for GroupDRO training of the final QFL scorer."""

from __future__ import annotations

import unittest

import pandas as pd
import torch

from dev.eval_actionness_qfl_groupdro_cv import fit_groupdro_qfl
from dev.eval_actionness_quality_head_cv import FEATURE_COLUMNS


class ActionnessQflGroupDroTest(unittest.TestCase):
    def test_fit_returns_normalized_recording_weights(self) -> None:
        rows = []
        for recording, target in (("easy", 0.0), ("hard", 1.0)):
            for index in range(4):
                row = {column: float(index) for column in FEATURE_COLUMNS}
                row.update(
                    {
                        "rec_name": recording,
                        "target_tiou": target,
                    }
                )
                rows.append(row)
        _, diagnostics = fit_groupdro_qfl(
            pd.DataFrame(rows),
            torch.device("cpu"),
            steps=2,
            learning_rate=0.01,
            eta=0.01,
        )
        self.assertEqual(set(diagnostics["rec_name"]), {"easy", "hard"})
        self.assertAlmostEqual(float(diagnostics["group_weight"].sum()), 1.0)

    def test_eta_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            fit_groupdro_qfl(
                pd.DataFrame(),
                torch.device("cpu"),
                steps=1,
                learning_rate=0.01,
                eta=0.0,
            )


if __name__ == "__main__":
    unittest.main()
