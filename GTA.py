import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time

# --- Configuration ---
SPACE_COOLDOWN = 0.3  # Min time (seconds) between space key presses

# --- Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
keyboard = Controller()

# Hand detection setup
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Global State
active_keys = {}  # Tracks currently held keys: {hand_label: Key_to_release}
last_space_press = 0  # Timestamp for debouncing the Space key

# --- Gesture Helper Functions ---

def is_finger_extended(landmarks, tip_id, pip_id):
    """Checks if a finger (Index, Middle, Ring, Pinky) is extended."""
    # A finger is extended if its tip (Y) is higher than its PIP joint (Y)
    return landmarks[tip_id].y < landmarks[pip_id].y

def is_thumb_extended(landmarks):
    """Special check for thumb extension (uses X-coordinate)."""
    # Compares the distance between the tip (4) and the MCP (2) vs the IP (3) and the MCP (2)
    # This is a robust way to check if the thumb is extended outward from the palm plane.
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    return abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x)

def detect_hand_state(landmarks):
    """Detects 'OPEN', 'CLOSED', or 'ONE_FINGER' gesture."""
    thumb_ext = is_thumb_extended(landmarks)
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext = is_finger_extended(landmarks, 16, 14)
    pinky_ext = is_finger_extended(landmarks, 20, 18)
    
    extended_fingers = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])
    
    # 1. ONE_FINGER (Index finger gun): Index extended, others (except maybe thumb) closed
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return 'ONE_FINGER'
    
    # 2. OPEN hand: 3 or more fingers extended
    if extended_fingers >= 3:
        return 'OPEN'
    
    # 3. CLOSED fist: 1 or fewer fingers extended (allowing for slight ambiguity/thumb)
    if extended_fingers <= 1:
        return 'CLOSED'
    
    return None # Ambiguous state

# --- Keyboard Action Handlers ---

def release_key(hand_label):
    """Releases the key currently held by this hand."""
    global active_keys
    
    if hand_label in active_keys:
        key_to_release = active_keys[hand_label]
        try:
            keyboard.release(key_to_release)
        except Exception:
            pass # Key might already be released
        del active_keys[hand_label]

def press_key(hand_label, key):
    """Presses the specified key, releasing any previously held key by this hand."""
    global active_keys
    
    # 1. Check if we need to release a conflicting key first
    if hand_label in active_keys and active_keys[hand_label] != key:
        release_key(hand_label) # Release conflicting key
    
    # 2. Press the new key if it's not already pressed
    if hand_label not in active_keys:
        try:
            keyboard.press(key)
            active_keys[hand_label] = key
        except Exception as e:
            print(f"Error pressing key {key}: {e}")

def single_press_key(key):
    """Performs a quick press/release with debouncing (used for Space)."""
    global last_space_press
    
    current_time = time.time()
    if current_time - last_space_press > SPACE_COOLDOWN:
        try:
            keyboard.press(key)
            keyboard.release(key)
            last_space_press = current_time
        except Exception as e:
            print(f"Error with single press: {e}")

def process_hand_gesture(hand_label, state):
    """Maps the detected gesture state to a keyboard command."""
    if hand_label == "Right":
        if state == 'OPEN':
            press_key(hand_label, Key.up)
        elif state == 'CLOSED':
            press_key(hand_label, Key.right)
        elif state == 'ONE_FINGER':
            # Single press action, do not hold the key
            single_press_key(Key.space)
            release_key(hand_label) # Ensure no key is held by this hand after tap
    
    elif hand_label == "Left":
        if state == 'OPEN':
            press_key(hand_label, Key.left)
        elif state == 'CLOSED':
            press_key(hand_label, Key.down)

# --- Main Application Loop ---

def main():
    """Starts the webcam feed and gesture recognition loop."""
    # Initialize video capture with preferred settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Hand Gesture Controller Active. Press 'q' or ESC in the video window to stop.")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to grab frame.")
                continue
            
            # Prepare frame: Flip for natural mirror view and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # --- Key and State Management ---
            current_frame_hands = {} # Tracks hands detected in this frame
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label # "Left" or "Right"
                    
                    # Draw visual landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Process the gesture
                    state = detect_hand_state(landmarks.landmark)
                    
                    if state:
                        current_frame_hands[hand_label] = state
                        process_hand_gesture(hand_label, state)
            
            # Release keys for hands that disappeared from the frame
            for hand_label in list(active_keys.keys()):
                if hand_label not in current_frame_hands:
                    release_key(hand_label)
            
            # --- Display Status on Video Feed ---
            
            cv2.putText(frame, "Gesture Controller Status:", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset = 70
            
            # Right Hand Status
            r_state = current_frame_hands.get("Right", "NOT DETECTED")
            r_key_map = {'OPEN': 'UP', 'CLOSED': 'RIGHT', 'ONE_FINGER': 'SPACE'}
            r_key = r_key_map.get(r_state, '')
            r_color = (0, 255, 255) if r_state != "NOT DETECTED" else (0, 0, 255)

            cv2.putText(frame, f"Right Hand: {r_state} -> {r_key}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2)
            y_offset += 35
            
            # Left Hand Status
            l_state = current_frame_hands.get("Left", "NOT DETECTED")
            l_key_map = {'OPEN': 'LEFT', 'CLOSED': 'DOWN'}
            l_key = l_key_map.get(l_state, '')
            l_color = (0, 255, 255) if l_state != "NOT DETECTED" else (0, 0, 255)

            cv2.putText(frame, f"Left Hand: {l_state} -> {l_key}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, l_color, 2)
            
            # Show the video feed
            cv2.imshow('Hand Gesture Controller', frame)
            
            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: # 'q' or ESC
                break
    
    finally:
        # Crucial Cleanup: release all held keys and resources
        for hand_label in list(active_keys.keys()):
            release_key(hand_label)
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nController stopped. All keys released.")

if __name__ == "__main__":
    main()