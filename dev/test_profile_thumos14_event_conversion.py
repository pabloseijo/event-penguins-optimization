"""Tests for the profiling helpers of the THUMOS14 event conversion.

Checks that the action window is centred and clamped to the video bounds and that
the selection key is stable and class-specific.
"""

from __future__ import annotations

import unittest

from dev.profile_thumos14_event_conversion import action_window, stable_key


class ProfileThumos14EventConversionTest(unittest.TestCase):
    def test_action_window_is_centered_and_clamped(self) -> None:
        self.assertEqual(action_window(20.0, 24.0, 100.0, 10.0), (17.0, 27.0))
        self.assertEqual(action_window(1.0, 3.0, 100.0, 10.0), (0.0, 10.0))
        self.assertEqual(action_window(98.0, 100.0, 100.0, 10.0), (90.0, 100.0))
        self.assertEqual(action_window(1.0, 3.0, 4.0, 10.0), (0.0, 4.0))

    def test_selection_key_is_stable_and_class_specific(self) -> None:
        candidate = {
            "video_id": "video_validation_0000001",
            "annotation_start_s": 1.0,
            "annotation_stop_s": 2.0,
        }
        self.assertEqual(stable_key(7, "LongJump", candidate), stable_key(7, "LongJump", candidate))
        self.assertNotEqual(stable_key(7, "LongJump", candidate), stable_key(7, "HighJump", candidate))


if __name__ == "__main__":
    unittest.main()
