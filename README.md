# MotionArcade
🕹️ MotionArcade: Gesture-Controlled Gaming
MotionArcade is an innovative Python project developed by Team Obsidian for our 1st semester, transforming traditional arcade gaming by allowing users to play games using hand gestures. We've created an arcade of programs where each game utilizes a specific program to map distinct hand gestures to different keyboard/mouse controls, providing an intuitive and immersive control experience.

- Team members = Churchil, Aditya, Granth, Dileep, Sachin.

- Works very well with python version 3.8.3.
  
✨ Features
Gesture-Based Control: Play games using natural hand movements instead of physical controllers.

Customizable Mapping: Different programs for different games, allowing unique gesture-to-control mapping (e.g., a "swipe" gesture might translate to a specific directional key in one game and a jump action in another).

Real-time Responsiveness: Utilizes efficient libraries for fast and accurate gesture recognition.

💻 Technology,Role in MotionArcade: 
OpenCV-Used for reading webcam input and processing video frames.
Mediapipe-Crucial for accurate detection of hand and pose landmarks (21 defined landmarks on the hand).
NumPy-"Utilized for calculations, coordinate scaling, and mapping webcam coordinates to screen coordinates."
Pynput-Enables the final step: simulating keyboard or mouse events based on the recognized gesture to control the game.

🚀 System Workflow
The platform operates through a series of real-time processing steps to translate a hand gesture into a game command:

Video Capture: A webcam records frames and instantly feeds them to the system.
Landmark Detection: Mediapipe detects the 21 key points (landmarks) of the hand.
Gesture Recognition: The system classifies the gesture (e.g., click, swipe, fist) by comparing the detected landmark positions.
Coordinate Mapping: NumPy scales and converts camera coordinates to screen coordinates for precise cursor or movement control.
Command Execution: Pynput generates the simulated input (keypress or mouse action) to control the active game.
