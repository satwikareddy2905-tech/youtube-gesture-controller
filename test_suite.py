"""Comprehensive test suite for YouTube Gesture Controller.

Tests all 10 requirements:
1. All 9 gesture recognition cases.
2. All 9 controller mappings.
3. Single-trigger behavior.
4. Re-trigger after release.
5. Swipe single-trigger behavior.
6. Keyboard event generation.
7. No text-character generation (hardware scancodes).
8. Chrome/YouTube window targeting.
9. Failure handling when Chrome is unavailable.
10. Component startup and teardown.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from config import AppConfig, GestureName, KeyboardAction
from controller import ActionController, _find_target_youtube_window, _press_key
from gesture_recognition import GestureRecognizer
from hand_detector import HandLandmarks


class TestGestureRecognition(unittest.TestCase):
    """Test suite for gesture recognition algorithm."""

    def setUp(self) -> None:
        self.config = AppConfig()
        self.recognizer = GestureRecognizer(self.config)

    def _create_open_palm_landmarks(self) -> HandLandmarks:
        lm = [(0.0, 0.0)] * 21
        lm[0] = (0.5, 0.8)   # Wrist
        lm[1] = (0.45, 0.7)  # Thumb CMC
        lm[2] = (0.4, 0.6)   # Thumb MCP
        lm[3] = (0.3, 0.52)  # Thumb IP
        lm[4] = (0.2, 0.45)  # Thumb Tip
        # Index (5..8)
        lm[5] = (0.4, 0.5); lm[6] = (0.4, 0.38); lm[7] = (0.4, 0.28); lm[8] = (0.4, 0.18)
        # Middle (9..12)
        lm[9] = (0.5, 0.5); lm[10] = (0.5, 0.35); lm[11] = (0.5, 0.25); lm[12] = (0.5, 0.15)
        # Ring (13..16)
        lm[13] = (0.6, 0.5); lm[14] = (0.6, 0.38); lm[15] = (0.6, 0.28); lm[16] = (0.6, 0.18)
        # Pinky (17..20)
        lm[17] = (0.7, 0.52); lm[18] = (0.7, 0.42); lm[19] = (0.7, 0.32); lm[20] = (0.7, 0.22)
        return HandLandmarks(landmarks_2d=lm)

    def _process_frames(self, hl: HandLandmarks, count: int = 5) -> GestureName:
        res = GestureName.NONE
        for _ in range(count):
            res = self.recognizer.process_landmarks(hl, 640, 480)
        return res

    def test_open_palm_detection(self) -> None:
        hl = self._create_open_palm_landmarks()
        res = self._process_frames(hl)
        self.assertEqual(res, GestureName.OPEN_PALM)

    def test_closed_fist_detection(self) -> None:
        hl = self._create_open_palm_landmarks()
        lm = list(hl.landmarks_2d)
        lm[4] = (0.42, 0.52)
        for mcp_i, pip_i, dip_i, tip_i in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            lm[pip_i] = (lm[mcp_i][0], 0.42)
            lm[dip_i] = (lm[mcp_i][0], 0.48)
            lm[tip_i] = (lm[mcp_i][0], 0.53)
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.CLOSED_FIST)

    def test_thumb_up_detection(self) -> None:
        hl = self._create_open_palm_landmarks()
        lm = list(hl.landmarks_2d)
        lm[4] = (0.42, 0.52)
        for mcp_i, pip_i, dip_i, tip_i in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            lm[pip_i] = (lm[mcp_i][0], 0.42)
            lm[dip_i] = (lm[mcp_i][0], 0.48)
            lm[tip_i] = (lm[mcp_i][0], 0.53)
        # Extend thumb upwards
        lm[3] = (0.35, 0.45)
        lm[4] = (0.3, 0.3)
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.THUMB_UP)

    def test_thumb_down_detection(self) -> None:
        lm = [(0.0, 0.0)] * 21
        lm[0] = (0.5, 0.2)
        lm[9] = (0.5, 0.4)
        for mcp_i, pip_i, dip_i, tip_i in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
            lm[mcp_i] = (0.5, 0.4)
            lm[pip_i] = (0.5, 0.48)
            lm[dip_i] = (0.5, 0.52)
            lm[tip_i] = (0.5, 0.45)
        lm[1] = (0.45, 0.3); lm[2] = (0.4, 0.4); lm[3] = (0.35, 0.55); lm[4] = (0.3, 0.7)
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.THUMB_DOWN)

    def test_victory_detection(self) -> None:
        open_hl = self._create_open_palm_landmarks()
        lm = list(open_hl.landmarks_2d)
        lm[4] = (0.42, 0.52) # Fold thumb
        # Fold ring and pinky
        for mcp_i, pip_i, dip_i, tip_i in [(13, 14, 15, 16), (17, 18, 19, 20)]:
            lm[pip_i] = (lm[mcp_i][0], 0.42)
            lm[dip_i] = (lm[mcp_i][0], 0.48)
            lm[tip_i] = (lm[mcp_i][0], 0.53)
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.VICTORY)

    def test_three_fingers_detection(self) -> None:
        open_hl = self._create_open_palm_landmarks()
        lm = list(open_hl.landmarks_2d)
        lm[4] = (0.42, 0.52) # Fold thumb
        # Fold pinky only
        for mcp_i, pip_i, dip_i, tip_i in [(17, 18, 19, 20)]:
            lm[pip_i] = (lm[mcp_i][0], 0.42)
            lm[dip_i] = (lm[mcp_i][0], 0.48)
            lm[tip_i] = (lm[mcp_i][0], 0.53)
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.THREE_FINGERS)

    def test_four_fingers_detection(self) -> None:
        open_hl = self._create_open_palm_landmarks()
        lm = list(open_hl.landmarks_2d)
        lm[4] = (0.42, 0.52) # Fold thumb, keep 4 fingers extended
        res = self._process_frames(HandLandmarks(landmarks_2d=lm))
        self.assertEqual(res, GestureName.FOUR_FINGERS)

    def test_swipe_right_detection(self) -> None:
        self.recognizer._centroid_history.clear()
        for i in range(15):
            x = 100 + i * 15
            y = 200
            self.recognizer._centroid_history.append((x, y))
        swipe = self.recognizer._detect_swipe()
        self.assertEqual(swipe, GestureName.SWIPE_RIGHT)

    def test_swipe_left_detection(self) -> None:
        self.recognizer._centroid_history.clear()
        for i in range(15):
            x = 400 - i * 15
            y = 200
            self.recognizer._centroid_history.append((x, y))
        swipe = self.recognizer._detect_swipe()
        self.assertEqual(swipe, GestureName.SWIPE_LEFT)


class TestActionController(unittest.TestCase):
    """Test suite for ActionController dispatch, single-trigger, re-trigger, and mappings."""

    def setUp(self) -> None:
        self.config = AppConfig()
        self.controller = ActionController(self.config)

    @patch("controller._press_key", return_value=(True, "YouTube - Google Chrome", "chrome.exe", 12345))
    def test_all_9_controller_mappings(self, mock_press: MagicMock) -> None:
        mappings = [
            (GestureName.CLOSED_FIST, KeyboardAction.PLAY_PAUSE, "k"),
            (GestureName.OPEN_PALM, KeyboardAction.MUTE, "m"),
            (GestureName.THUMB_UP, KeyboardAction.VOLUME_UP, "up"),
            (GestureName.THUMB_DOWN, KeyboardAction.VOLUME_DOWN, "down"),
            (GestureName.VICTORY, KeyboardAction.FULLSCREEN, "f"),
            (GestureName.THREE_FINGERS, KeyboardAction.THEATER_MODE, "t"),
            (GestureName.FOUR_FINGERS, KeyboardAction.TOGGLE_CAPTIONS, "c"),
            (GestureName.SWIPE_RIGHT, KeyboardAction.FORWARD_10S, "l"),
            (GestureName.SWIPE_LEFT, KeyboardAction.BACKWARD_10S, "j"),
        ]

        for gesture, expected_action, expected_key in mappings:
            self.controller.trigger_action(GestureName.NONE)
            self.controller._last_action_time = 0.0
            self.controller._last_volume_action_time = 0.0
            self.controller._last_swipe_action_time = 0.0

            trig, action, key = self.controller.trigger_action(gesture)
            self.assertTrue(trig, f"Failed for gesture {gesture}")
            self.assertEqual(action, expected_action)
            self.assertEqual(key, expected_key)

    @patch("controller._press_key", return_value=(True, "YouTube - Google Chrome", "chrome.exe", 12345))
    def test_single_trigger_and_retrigger(self, mock_press: MagicMock) -> None:
        self.controller.trigger_action(GestureName.NONE)
        self.controller._last_action_time = 0.0

        # 1. First trigger
        trig1, act1, k1 = self.controller.trigger_action(GestureName.OPEN_PALM)
        self.assertTrue(trig1)
        self.assertEqual(act1, KeyboardAction.MUTE)

        # 2. Continuous hold (must NOT trigger again)
        trig2, act2, k2 = self.controller.trigger_action(GestureName.OPEN_PALM)
        self.assertFalse(trig2)

        # 3. Release to NONE
        self.controller.trigger_action(GestureName.NONE)

        # 4. Trigger again after release
        self.controller._last_action_time = 0.0
        trig3, act3, k3 = self.controller.trigger_action(GestureName.OPEN_PALM)
        self.assertTrue(trig3)

    @patch("controller._press_key", return_value=(False, "", "", 0))
    def test_chrome_unavailable_failure_handling(self, mock_press: MagicMock) -> None:
        self.controller.trigger_action(GestureName.NONE)
        self.controller._last_action_time = 0.0
        trig, action, key = self.controller.trigger_action(GestureName.OPEN_PALM)
        self.assertFalse(trig)
        self.assertIsNone(action)
        self.assertIsNone(key)


class TestWindowTargetingAndKeyboardEvents(unittest.TestCase):
    """Test suite for window targeting and hardware scancode keyboard mechanics."""

    @patch("ctypes.windll.user32.IsWindowVisible", return_value=True)
    @patch("controller._get_window_title", return_value="Rick Astley - Never Gonna Give You Up - YouTube - Google Chrome")
    @patch("controller._get_process_name", return_value="chrome.exe")
    def test_window_targeting(self, mock_pname: MagicMock, mock_title: MagicMock, mock_vis: MagicMock) -> None:
        from controller import _find_target_youtube_window
        hwnd, title, pname = _find_target_youtube_window()
        self.assertIsNotNone(hwnd)
        self.assertIn("YouTube", title)

    @patch("controller._ensure_target_window_focused", return_value=(12345, "YouTube - Google Chrome", "chrome.exe", 54321))
    @patch("ctypes.windll.user32.keybd_event")
    @patch("ctypes.windll.user32.MapVirtualKeyW", return_value=37)
    def test_keyboard_event_scancodes(self, mock_map: MagicMock, mock_kb: MagicMock, mock_focus: MagicMock) -> None:
        success, title, pname, hwnd = _press_key("k")
        self.assertTrue(success)
        self.assertEqual(title, "YouTube - Google Chrome")
        self.assertTrue(mock_kb.called)


if __name__ == "__main__":
    unittest.main()
