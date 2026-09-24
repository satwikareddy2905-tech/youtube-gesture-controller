## YouTube Gesture Controller

A real-time computer vision application that allows users to control YouTube videos using hand gestures.

The system uses a webcam to detect hand movements, recognizes predefined gestures using MediaPipe, and converts them into YouTube controls through keyboard automation.

---

##  Overview

The **YouTube Gesture Controller** provides a touch-free way to interact with YouTube.

Instead of using a keyboard or mouse, users can perform simple hand gestures in front of a webcam to control video playback, volume, fullscreen, captions, theater mode, and video seeking.

The project combines:

- Computer Vision
- Hand Landmark Detection
- Gesture Recognition
- Real-Time Processing
- Windows Keyboard Automation

---

##  Problem Statement

Traditional video controls require users to interact physically with a keyboard or mouse.

This can be inconvenient when:

- The user is away from the keyboard
- The keyboard or mouse is not easily accessible
- Touch-free interaction is preferred
- A hands-free demonstration environment is required

This project provides an alternative interaction method using hand gestures.



##  Proposed Solution

The system captures the user's hand through a webcam and processes the video using MediaPipe Hand Landmarker.

The detected hand landmarks are analyzed to identify predefined gestures.

The recognized gesture is then converted into the corresponding YouTube control.

### Workflow

```text
Webcam
   ↓
Video Capture
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Gesture Recognition
   ↓
Gesture Stabilization
   ↓
Controller
   ↓
Keyboard Event
   ↓
YouTube
```

### output

<img width="1603" height="981" alt="image" src="https://github.com/user-attachments/assets/2d4dfa61-b9e1-496e-beb2-a932d460c3d7" />

<img width="1584" height="993" alt="image" src="https://github.com/user-attachments/assets/4d31adea-76a8-4b04-9ad1-d58490f565b0" />

