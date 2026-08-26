"""Tests for the output-directory guard of the final transfer run.

Checks that a missing directory is created, that an existing empty one is
accepted, and that an existing artifact is never overwritten.
"""

import tempfile
import unittest
from pathlib import Path

from run_actionformer_transfer_final import create_empty_output_dir


class RunActionFormerTransferFinalTest(unittest.TestCase):
    def test_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final"

            create_empty_output_dir(output)

            self.assertTrue(output.is_dir())

    def test_allows_existing_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            create_empty_output_dir(output)

            self.assertTrue(output.is_dir())

    def test_refuses_to_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "predictions.npz").write_bytes(b"frozen")

            with self.assertRaises(FileExistsError):
                create_empty_output_dir(output)


if __name__ == "__main__":
    unittest.main()
