import math
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time

                                   
# --- I nitialization ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
keyboard = Controller() # Initialize virtual keyboard
     
# --- Constants for Gesture ---
# Normalized distance threshold for a 'pinch' gesture (Thumb Tip to Index Tip)
# TUNE THIS VALUE: Start with 0 .1, increase if too sensitive, decrease if too hard to trigger.
FLAP_THRESHOLD = 0.2
# Cooldown to prevent spamming the Spacebar from a single gesture
FLAP_COOLDOWN = 0.3 #seconds

# --- Tracking Variables ---
last_flap_time = 0.0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased confidence for stability
    min_tracking_confidence=0.5) as hands:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    current_time = time.time()
    
    # Process the image
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        # 1. Get Thumb Tip (4) and Index Finger Tip (8)
        l = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        r = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # 2. Calculate Normalized Distance
        dist = math.sqrt((l.x - r.x)**2 + (l.y - r.y)**2)
        
        # 3. Gesture Logic: Check for Pinch and Cooldown
        if dist < FLAP_THRESHOLD and (current_time - last_flap_time) > FLAP_COOLDOWN:
          
          # Inject Keyboard Input (The Flap action!)
          keyboard.press(Key.space)
          keyboard.release(Key.space)
          
          last_flap_time = current_time # Reset Cooldown
          # cv2.putText(image, "FLAP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          # # print("SPACE")

        # Draw Landmarks
        mp_drawing.draw_landmarks(                       
            image,
            hand_landmarks,   
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Hand Gesture Input Injector', cv2.flip(image, 1))
    
    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27: 
      break

cap.release()
cv2.destroyAllWindows()