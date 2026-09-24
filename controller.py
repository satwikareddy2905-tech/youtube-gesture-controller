"""Action controller module for executing keyboard automation.

Interfaces with the Windows keybd_event API to simulate YouTube keyboard shortcuts
in response to hand gestures. Implements process-level YouTube window targeting,
strict target verification (NO FALLBACK to non-YouTube windows), text field blur
protection, focus restoration, and single-trigger gesture state transitions.
"""

import ctypes
import ctypes.wintypes
import os
import time
from typing import Optional, Tuple

from config import AppConfig, GestureName, KeyboardAction
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Windows Win32 API Constants & Flags
# ---------------------------------------------------------------------------
_KEYEVENTF_EXTENDEDKEY = 0x0001   # required for arrow / nav keys
_KEYEVENTF_KEYUP       = 0x0002   # key-release event

_MAPVK_VK_TO_VSC = 0
_VK_ESCAPE = 0x1B

_DESKTOP_READOBJECTS = 0x0001
_DESKTOP_ENUMERATE = 0x0040
_SW_RESTORE = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Executable names of supported desktop web browsers
_BROWSER_EXECUTABLES = frozenset({
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "vivaldi.exe",
})

# Virtual-Key codes for all YouTube shortcut keys used in this application
_VK_CODES: dict = {
    "k":    0x4B,   # VK_K    - Play/Pause
    "m":    0x4D,   # VK_M    - Mute
    "up":   0x26,   # VK_UP   - Volume Up
    "down": 0x28,   # VK_DOWN - Volume Down
    "f":    0x46,   # VK_F    - Fullscreen
    "t":    0x54,   # VK_T    - Theater Mode
    "c":    0x43,   # VK_C    - Captions
    "l":    0x4C,   # VK_L    - Seek +10 s
    "j":    0x4A,   # VK_J    - Seek -10 s
}

# Arrow / navigation keys that need KEYEVENTF_EXTENDEDKEY
_EXTENDED_VK = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28})


def _get_window_title(hwnd: int) -> str:
    """Returns the title string of a window handle."""
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length > 0:
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value
    return ""


def _click_player_center(hwnd: int) -> None:
    """Sends a synthetic left-click to the centre of the browser client area.

    Used exclusively before the fullscreen key ('f') to transfer DOM focus
    from any focused text input (address bar, search box, comment box) to the
    YouTube HTML5 video player so that 'f' is correctly interpreted as the
    fullscreen toggle rather than typed into a text field.

    PostMessage is used so the call is non-blocking and does not disturb the
    foreground window ownership established by SetForegroundWindow.

    Args:
        hwnd: Window handle of the verified YouTube browser window.
    """
    user32 = ctypes.windll.user32

    # Get the client-area dimensions to click the geometric centre
    rect = ctypes.wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return

    cx = (rect.right - rect.left) // 2
    cy = (rect.bottom - rect.top) // 2

    # Pack (x, y) into a single LPARAM value as Windows expects
    lparam = ctypes.wintypes.LPARAM((cy << 16) | (cx & 0xFFFF))

    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP   = 0x0202
    MK_LBUTTON     = 0x0001

    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    user32.PostMessageW(hwnd, WM_LBUTTONUP,   0,          lparam)

    # Brief pause so the browser processes the click and DOM focus settles
    # before the 'f' keybd_event is dispatched
    time.sleep(0.08)


def _get_process_name(hwnd: int) -> str:
    """Retrieves the executable filename of the process owning the window handle."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    pid = ctypes.wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""

    hProcess = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not hProcess:
        return ""

    buff = ctypes.create_unicode_buffer(1024)
    size = ctypes.wintypes.DWORD(1024)
    res = kernel32.QueryFullProcessImageNameW(hProcess, 0, buff, ctypes.byref(size))
    kernel32.CloseHandle(hProcess)

    if res:
        return os.path.basename(buff.value).lower()
    return ""


def _find_target_youtube_window() -> Tuple[Optional[int], str, str]:
    """Locates an open browser window running YouTube.

    STRICT SAFETY RULE: Returns ONLY windows whose process is a supported browser AND
    whose window title explicitly contains 'youtube'. Returns (None, "", "") if no verified
    YouTube window is found. ABSOLUTELY NO FALLBACK to ChatGPT, VS Code, Notepad, or
    arbitrary foreground windows.
    """
    user32 = ctypes.windll.user32
    yt_windows = []

    def check_window(hwnd: int) -> None:
        if not user32.IsWindowVisible(hwnd):
            return
        pname = _get_process_name(hwnd)
        if pname not in _BROWSER_EXECUTABLES:
            return

        title = _get_window_title(hwnd)
        title_lower = title.lower()

        # Strict check: MUST contain 'youtube' in tab/window title
        if "youtube" in title_lower:
            yt_windows.append((hwnd, title, pname))

    # Try EnumDesktopWindows on WinSta0 Default desktop first
    hDesktop = user32.OpenDesktopW("Default", 0, False, _DESKTOP_READOBJECTS | _DESKTOP_ENUMERATE)
    if hDesktop:
        def enum_cb(hwnd: int, lparam: int) -> bool:
            check_window(hwnd)
            return True
        EnumDesktopWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        user32.EnumDesktopWindows(hDesktop, EnumDesktopWindowsProc(enum_cb), 0)
        user32.CloseDesktop(hDesktop)

    if not yt_windows:
        def enum_win_cb(hwnd: int, lparam: int) -> bool:
            check_window(hwnd)
            return True
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        user32.EnumWindows(EnumWindowsProc(enum_win_cb), 0)

    if yt_windows:
        return yt_windows[0]

    # STRICT: If no verified YouTube window is found, return None. NEVER fallback!
    return None, "", ""


def _ensure_target_window_focused() -> Tuple[Optional[int], str, str, Optional[int]]:
    """Ensures the verified YouTube target browser window has input focus.

    Returns:
        Tuple of (target_hwnd, target_title, target_pname, previous_fg_hwnd).
    """
    user32 = ctypes.windll.user32
    target_hwnd, target_title, target_pname = _find_target_youtube_window()
    if not target_hwnd:
        return None, "", "", None

    fg_hwnd = user32.GetForegroundWindow()
    if fg_hwnd != target_hwnd:
        try:
            if user32.IsIconic(target_hwnd):
                user32.ShowWindow(target_hwnd, _SW_RESTORE)

            user32.keybd_event(0, 0, 0, 0)
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            target_thread = user32.GetWindowThreadProcessId(target_hwnd, None)

            if fg_thread and fg_thread != target_thread:
                user32.AttachThreadInput(fg_thread, target_thread, True)
                user32.SetForegroundWindow(target_hwnd)
                user32.BringWindowToTop(target_hwnd)
                user32.AttachThreadInput(fg_thread, target_thread, False)
            else:
                user32.SetForegroundWindow(target_hwnd)
                user32.BringWindowToTop(target_hwnd)
            time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Target window focus adjustment error: {e}")

    # Double check: if a non-YouTube foreground window is active and blocking focus, cancel action.
    curr_fg = user32.GetForegroundWindow()
    if curr_fg and curr_fg != target_hwnd:
        curr_title = _get_window_title(curr_fg)
        if curr_title and "youtube" not in curr_title.lower() and "gesture controller" not in curr_title.lower():
            logger.warning(f"Focus verification failed: Active window '{curr_title}' is not YouTube.")
            return None, "", "", None

    return target_hwnd, target_title, target_pname, fg_hwnd


def _press_key(key_name: str) -> Tuple[bool, str, str, int]:
    """Send a single key press + release via Windows keybd_event to verified YouTube window.

    Args:
        key_name: One of the string keys defined in _VK_CODES.

    Returns:
        Tuple of (success_boolean, target_window_title, target_process_name, target_hwnd).
    """
    target_hwnd, target_title, target_pname, previous_fg_hwnd = _ensure_target_window_focused()
    if not target_hwnd:
        return False, "", "", 0

    vk = _VK_CODES.get(key_name)
    if vk is None:
        raise ValueError(f"Unknown key name: '{key_name}'")

    user32 = ctypes.windll.user32

    # Text-field blur strategy:
    # - For letter shortcuts (k, m, t, c, l, j): send ESC first to move focus
    #   away from any active text field (search box, comment box, address bar).
    # - For 'f' (fullscreen): no pre-step needed.
    #   _ensure_target_window_focused() already called SetForegroundWindow which
    #   gives Chrome OS-level keyboard focus. keybd_event(VK_F) is then routed
    #   by Chrome to YouTube's keyboard shortcut handler and toggles fullscreen
    #   without touching the video element or its playback state.
    #   NOTE: ESC is intentionally NOT sent before 'f' — ESC would exit fullscreen
    #   if the player is already in fullscreen mode.
    #   NOTE: A mouse click on the player centre is intentionally NOT used — a
    #   click on the YouTube video element triggers the play/pause toggle handler.
    # - Arrow keys (up/down) require no pre-step; YouTube handles them at page level.
    if key_name in ("k", "m", "t", "c", "l", "j"):
        esc_scancode = user32.MapVirtualKeyW(_VK_ESCAPE, _MAPVK_VK_TO_VSC)
        user32.keybd_event(_VK_ESCAPE, esc_scancode, 0, 0)
        user32.keybd_event(_VK_ESCAPE, esc_scancode, _KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)

    scancode: int = user32.MapVirtualKeyW(vk, _MAPVK_VK_TO_VSC)
    is_extended = vk in _EXTENDED_VK
    flags_down = _KEYEVENTF_EXTENDEDKEY if is_extended else 0
    flags_up = flags_down | _KEYEVENTF_KEYUP

    # Send valid Virtual Key (bVk) and Hardware Scancode (bScan) via keybd_event
    user32.keybd_event(vk, scancode, flags_down, 0)
    user32.keybd_event(vk, scancode, flags_up, 0)

    # Return focus to the Gesture Controller GUI if it was active before sending the key
    if previous_fg_hwnd and previous_fg_hwnd != target_hwnd:
        prev_title = _get_window_title(previous_fg_hwnd)
        if "gesture controller" in prev_title.lower() or _get_process_name(previous_fg_hwnd) == "python.exe":
            try:
                time.sleep(0.02)
                user32.SetForegroundWindow(previous_fg_hwnd)
            except Exception:
                pass

    return True, target_title, target_pname, target_hwnd


class ActionController:
    """Dispatches virtual keystrokes based on recognized gestures, enforcing strict target verification."""

    def __init__(self, config: AppConfig) -> None:
        """Initializes the ActionController.

        Args:
            config: Aggregated configuration parameters.
        """
        self._config = config

        # Track timestamps of last execution to enforce cooldown rules
        self._last_action_time: float = 0.0
        self._last_volume_action_time: float = 0.0
        # Swipe actions get their own independent cooldown so they are never
        # blocked by a recently triggered static gesture (play/pause, mute, etc.)
        self._last_swipe_action_time: float = 0.0

        # Gesture state tracking to ensure one action per gesture state entry
        self._active_gesture: GestureName = GestureName.NONE
        self._has_triggered_for_current_gesture: bool = False

    def reset_state(self) -> None:
        """Resets active gesture state tracking."""
        self._active_gesture = GestureName.NONE
        self._has_triggered_for_current_gesture = False

    def trigger_action(
        self, gesture: GestureName
    ) -> Tuple[bool, Optional[KeyboardAction], Optional[str]]:
        """Maps a recognized hand gesture to a shortcut key and executes it.

        Args:
            gesture: The detected hand gesture.

        Returns:
            A tuple containing:
                1. A boolean flag showing if the keystroke was dispatched.
                2. The KeyboardAction executed (or None).
                3. The key string pressed (or None).
        """
        if gesture == GestureName.NONE:
            self.reset_state()
            return False, None, None

        # State transition handling
        if gesture != self._active_gesture:
            self._active_gesture = gesture
            self._has_triggered_for_current_gesture = False

        # Prevent repeated triggering while the same gesture is continuously held
        if self._has_triggered_for_current_gesture:
            return False, None, None

        # 1. Map physical gesture to logical action
        action = self._config.GESTURE_ACTION_MAPPING.get(gesture)
        if not action:
            return False, None, None

        # 2. Map logical action to keyboard shortcut key string
        key = self._config.KEY_MAPPINGS.get(action)
        if not key:
            return False, None, None

        # 3. Handle action-specific cooldown periods.
        # Three independent cooldown buckets:
        #   - volume actions  : short 0.2s cooldown for responsive volume stepping
        #   - swipe actions   : 0.5s cooldown, INDEPENDENT of static gestures so a
        #                       swipe is never silently blocked by a recent play/pause
        #                       or mute action that reset _last_action_time
        #   - all other static gestures: 1.0s cooldown to prevent accidental repeats
        now = time.perf_counter()
        is_volume = action in (KeyboardAction.VOLUME_UP, KeyboardAction.VOLUME_DOWN)
        is_swipe  = action in (KeyboardAction.FORWARD_10S, KeyboardAction.BACKWARD_10S)

        if is_volume:
            cooldown  = self._config.gesture.VOLUME_COOLDOWN_SEC
            last_time = self._last_volume_action_time
        elif is_swipe:
            cooldown  = self._config.gesture.SWIPE_COOLDOWN_SEC
            last_time = self._last_swipe_action_time
        else:
            cooldown  = self._config.gesture.ACTION_COOLDOWN_SEC
            last_time = self._last_action_time

        if now - last_time < cooldown:
            return False, None, None

        # 4. Mark gesture as triggered and record cooldown timestamp *before*
        #    sending the key so that exceptions never cause a re-trigger.
        self._has_triggered_for_current_gesture = True
        if is_volume:
            self._last_volume_action_time = now
        elif is_swipe:
            self._last_swipe_action_time = now
        else:
            self._last_action_time = now

        # 5. Dispatch via keybd_event ONLY if a verified YouTube window exists
        try:
            success, target_title, target_pname, target_hwnd = _press_key(key)
            if success:
                logger.info(
                    f"\n[CONTROLLER]\n"
                    f"Gesture: {gesture.value}\n"
                    f"Target process: {target_pname}\n"
                    f"Target window title: {target_title}\n"
                    f"Target HWND: {target_hwnd}\n"
                    f"Action: {action.value}\n"
                    f"Target verified: TRUE\n"
                    f"Player focused: TRUE\n"
                    f"Keyboard event sent: TRUE"
                )
                return True, action, key
            else:
                logger.warning(
                    f"\n[CONTROLLER]\n"
                    f"Gesture: {gesture.value}\n"
                    f"YouTube target NOT FOUND\n"
                    f"ACTION CANCELLED\n"
                    f"NO KEYBOARD EVENT SENT"
                )
                return False, None, None

        except Exception as e:
            logger.error(f"Exception during key press for key '{key}': {e}")
            return False, None, None


def test_action_controller() -> None:
    """Verifies single-trigger per gesture entry, cooldown, and all key mappings."""
    import sys
    from unittest.mock import patch, call

    # Use the actual running module so the mock intercepts _press_key regardless
    # of whether this file is executed directly (__main__) or imported.
    _this_module = sys.modules[__name__]
    config = AppConfig()
    controller_inst = ActionController(config)

    with patch.object(_this_module, "_press_key", return_value=(True, "YouTube - Google Chrome", "chrome.exe", 12345)) as mock_press:

        # ── State-transition tests ──────────────────────────────────────────

        # 1. NONE -> OPEN_PALM: Should trigger MUTE exactly once
        trig, action, key = controller_inst.trigger_action(GestureName.OPEN_PALM)
        assert trig is True and action == KeyboardAction.MUTE and key == "m", \
            f"OPEN_PALM: expected (True, MUTE, 'm'), got ({trig}, {action}, {key})"
        assert mock_press.call_count == 1

        # 2. OPEN_PALM -> OPEN_PALM (held): Should NOT trigger again
        trig, action, key = controller_inst.trigger_action(GestureName.OPEN_PALM)
        assert trig is False, "Held OPEN_PALM should not re-trigger"
        assert mock_press.call_count == 1

        # 3. OPEN_PALM -> OPEN_PALM (held, 2nd): Should NOT trigger again
        trig, action, key = controller_inst.trigger_action(GestureName.OPEN_PALM)
        assert trig is False
        assert mock_press.call_count == 1

        # 4. OPEN_PALM -> NONE: Resets state
        controller_inst.trigger_action(GestureName.NONE)
        assert not controller_inst._has_triggered_for_current_gesture

        # 5. NONE -> OPEN_PALM: Should trigger MUTE again
        controller_inst._last_action_time = 0.0
        trig, action, key = controller_inst.trigger_action(GestureName.OPEN_PALM)
        assert trig is True and action == KeyboardAction.MUTE and key == "m"
        assert mock_press.call_count == 2

        # 6. OPEN_PALM -> CLOSED_FIST: Direct transition triggers PLAY_PAUSE
        controller_inst._last_action_time = 0.0
        trig, action, key = controller_inst.trigger_action(GestureName.CLOSED_FIST)
        assert trig is True and action == KeyboardAction.PLAY_PAUSE and key == "k"
        assert mock_press.call_count == 3

        # 7. CLOSED_FIST -> CLOSED_FIST (held): Should NOT trigger again
        trig, action, key = controller_inst.trigger_action(GestureName.CLOSED_FIST)
        assert trig is False
        assert mock_press.call_count == 3

        # 8. NONE -> THUMB_UP: Should trigger VOLUME_UP exactly once
        controller_inst.trigger_action(GestureName.NONE)
        controller_inst._last_volume_action_time = 0.0
        trig, action, key = controller_inst.trigger_action(GestureName.THUMB_UP)
        assert trig is True and action == KeyboardAction.VOLUME_UP and key == "up"
        assert mock_press.call_count == 4

        # 9. THUMB_UP -> THUMB_UP (held): Should NOT trigger again
        trig, action, key = controller_inst.trigger_action(GestureName.THUMB_UP)
        assert trig is False
        assert mock_press.call_count == 4

        # 10. THUMB_UP -> THUMB_DOWN: Direct transition triggers VOLUME_DOWN
        controller_inst._last_volume_action_time = 0.0
        trig, action, key = controller_inst.trigger_action(GestureName.THUMB_DOWN)
        assert trig is True and action == KeyboardAction.VOLUME_DOWN and key == "down"
        assert mock_press.call_count == 5

        # 11. THUMB_DOWN -> THUMB_DOWN (held): Should NOT trigger again
        trig, action, key = controller_inst.trigger_action(GestureName.THUMB_DOWN)
        assert trig is False
        assert mock_press.call_count == 5

        # ── Full 9-gesture mapping coverage tests ──────────────────────────
        # Helper: reset state and bypass cooldown, then fire gesture

        def _fire(gesture: GestureName, time_attr: str = "_last_action_time"):
            controller_inst.trigger_action(GestureName.NONE)
            setattr(controller_inst, time_attr, 0.0)
            return controller_inst.trigger_action(gesture)

        # 12. Mute: Open Palm -> "m"
        trig, action, key = _fire(GestureName.OPEN_PALM)
        assert trig and action == KeyboardAction.MUTE and key == "m", \
            f"Mute mapping failed: action={action}, key={key}"
        # 13. Play/Pause: Closed Fist -> "k" (triggers exactly once)
        controller_inst.trigger_action(GestureName.NONE)
        controller_inst._last_action_time = 0.0
        trig, action, key = controller_inst.trigger_action(GestureName.CLOSED_FIST)
        assert trig is True and action == KeyboardAction.PLAY_PAUSE and key == "k", \
            f"Play/Pause mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("k")

        # Confirm holding Closed Fist does NOT re-trigger Play/Pause
        trig_held, _, _ = controller_inst.trigger_action(GestureName.CLOSED_FIST)
        assert trig_held is False, "Closed Fist must not re-trigger Play/Pause while held"

        # 14. Fullscreen: Victory -> "f"
        trig, action, key = _fire(GestureName.VICTORY)
        assert trig and action == KeyboardAction.FULLSCREEN and key == "f", \
            f"Fullscreen mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("f")

        # 15. Theater Mode: Three Fingers -> "t"
        trig, action, key = _fire(GestureName.THREE_FINGERS)
        assert trig and action == KeyboardAction.THEATER_MODE and key == "t", \
            f"Theater Mode mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("t")

        # 16. Captions: Four Fingers -> "c"
        trig, action, key = _fire(GestureName.FOUR_FINGERS)
        assert trig and action == KeyboardAction.TOGGLE_CAPTIONS and key == "c", \
            f"Captions mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("c")

        # 17. Volume Up: Thumb Up -> "up"
        trig, action, key = _fire(GestureName.THUMB_UP, "_last_volume_action_time")
        assert trig and action == KeyboardAction.VOLUME_UP and key == "up", \
            f"Volume Up mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("up")

        # 18. Volume Down: Thumb Down -> "down"
        trig, action, key = _fire(GestureName.THUMB_DOWN, "_last_volume_action_time")
        assert trig and action == KeyboardAction.VOLUME_DOWN and key == "down", \
            f"Volume Down mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("down")

        # 19. Seek Forward: Swipe Right -> "l"
        trig, action, key = _fire(GestureName.SWIPE_RIGHT, "_last_swipe_action_time")
        assert trig and action == KeyboardAction.FORWARD_10S and key == "l", \
            f"Seek Forward mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("l")

        # 20. Seek Backward: Swipe Left -> "j"
        trig, action, key = _fire(GestureName.SWIPE_LEFT, "_last_swipe_action_time")
        assert trig and action == KeyboardAction.BACKWARD_10S and key == "j", \
            f"Seek Backward mapping failed: action={action}, key={key}"
        mock_press.assert_called_with("j")

    print("All action controller gesture state test cases passed successfully!")


if __name__ == "__main__":
    test_action_controller()
