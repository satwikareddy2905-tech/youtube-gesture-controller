"""Safety test verifying zero keyboard events sent when ChatGPT/non-YouTube is active."""

import unittest
from unittest.mock import MagicMock, patch

from config import AppConfig, GestureName
from controller import ActionController, _find_target_youtube_window


class TestChatGPTTargetingSafety(unittest.TestCase):
    """Test suite ensuring non-YouTube windows (ChatGPT, VSCode, etc.) never receive key events."""

    @patch("controller._get_process_name", return_value="chrome.exe")
    @patch("controller._get_window_title", return_value="ChatGPT - Google Chrome")
    @patch("ctypes.windll.user32.IsWindowVisible", return_value=True)
    def test_chatgpt_window_ignored(
        self, mock_vis: MagicMock, mock_title: MagicMock, mock_pname: MagicMock
    ) -> None:
        hwnd, title, pname = _find_target_youtube_window()
        self.assertIsNone(hwnd, "ChatGPT window must NOT be selected as target!")
        self.assertEqual(title, "")
        self.assertEqual(pname, "")

    @patch("ctypes.windll.user32.keybd_event")
    def test_action_cancelled_when_youtube_missing(self, mock_kb: MagicMock) -> None:
        config = AppConfig()
        controller = ActionController(config)
        controller.reset_state()

        with patch("controller._find_target_youtube_window", return_value=(None, "", "")):
            trig, action, key = controller.trigger_action(GestureName.THUMB_UP)
            self.assertFalse(trig, "Action must be CANCELLED when YouTube target missing!")
            self.assertIsNone(action)
            self.assertIsNone(key)
            self.assertFalse(mock_kb.called, "keybd_event MUST NEVER BE CALLED when YouTube target missing!")


if __name__ == "__main__":
    unittest.main()
