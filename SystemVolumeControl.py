import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# Maximum angle between fingers and maximum volume level
max_angle = 180.0
max_volume = 0.7

# Get the default audio output device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_control = cast(interface, POINTER(IAudioEndpointVolume))

def calculate_angle(a, b, c):
    """Calculate the angle between three points (in radians)."""
    radians = abs(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    return np.degrees(radians)

def set_system_volume(angle):
    """Set the system volume based on the finger angle."""
    volume = min(max(angle / max_angle, 0.0), 1.0)  # Normalize angle to range [0, 1]

    # Adjust volume smoothly
    current_volume = volume_control.GetMasterVolumeLevelScalar()
    volume_delta = 0.05  # Change in volume per frame
    if volume > current_volume:
        volume = min(current_volume + volume_delta, volume)
    elif volume < current_volume:
        volume = max(current_volume - volume_delta, volume)

    # Set the system volume level
    try:
        volume_control.SetMasterVolumeLevelScalar(volume, None)
    except Exception as e:
        print(f"Failed to set volume: {e}")

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and a blue line between thumb and index finger on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for thumb, index, and middle fingers
            landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            thumb_tip, index_tip, middle_tip = landmarks[4], landmarks[8], landmarks[12]

            # Calculate the angle between fingers and set the volume
            angle = calculate_angle(thumb_tip, index_tip, middle_tip)
            set_system_volume(angle)

            # Draw blue line between thumb and index finger
            cv2.line(frame, (int(thumb_tip[0] * frame.shape[1]), int(thumb_tip[1] * frame.shape[0])),
                     (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])),
                     (255, 0, 0), 3)

            # Display the angle on the frame
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
