import cv2
import numpy as np
import tensorflow as tf
from collections import deque  # <--- NEW: This is for our "Voting Buffer"

# 1. Load your trained "Brain"
model = tf.keras.models.load_model(
    'face_shape_model.keras') 

class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- NEW: Create a memory buffer that remembers the last 15 guesses ---
prediction_history = deque(maxlen=15)
stable_shape = "Detecting..."

print("Press 'q' to close the window.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Get frame dimensions so our padding doesn't accidentally go off the screen
    frame_height, frame_width, _ = frame.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    for (x, y, w, h) in faces:

        # --- FIX 1: PADDING (Expand the box by 25% so we don't cut off the jaw/hair) ---
        margin_w = int(w * 0.25)
        margin_h = int(h * 0.25)

        # Make sure our expanded box doesn't go off the edge of the camera screen
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(frame_width, x + w + margin_w)
        y2 = min(frame_height, y + h + margin_h)

        # Crop using the new expanded box
        face_roi = frame[y1:y2, x1:x2]
        # -------------------------------------------------------------------------

        try:
            # Resize and Predict
            resized_face = cv2.resize(face_roi, (224, 224))
            img_array = tf.keras.utils.img_to_array(resized_face)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            predicted_shape = class_names[np.argmax(score)]

            # --- FIX 2: THE VOTING SYSTEM ---
            # Add this split-second guess to our memory
            prediction_history.append(predicted_shape)

            # Find out which shape won the "vote" in the last 15 frames
            stable_shape = max(set(prediction_history),
                               key=prediction_history.count)
            # --------------------------------

            # Write the STABLE shape on the screen
            text = f"Shape: {stable_shape}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            pass

        # Keep the blue box and green dot
        forehead_x = x + (w // 2)
        forehead_y = y + int(h * 0.15)

        cv2.circle(frame, (forehead_x, forehead_y), 8, (0, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Virtual Mirror - AI Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
