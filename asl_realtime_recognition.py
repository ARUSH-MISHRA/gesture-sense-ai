import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model

model = load_model('asl_realphoto_cnn.h5')
with open('realphoto_labels.json', 'r') as f:
    labels = json.load(f)

IMG_SIZE = 64  # must match training size

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        box_w, box_h = x_max - x_min, y_max - y_min
        size = max(box_w, box_h, 1)
        padding = 40
        size += padding * 2

        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        x_min = max(0, cx - size // 2)
        x_max = min(w, cx + size // 2)
        y_min = max(0, cy - size // 2)
        y_max = min(h, cy + size // 2)

        hand_img = frame[y_min:y_max, x_min:x_max]

        if hand_img.size > 0 and hand_img.shape[0] > 5 and hand_img.shape[1] > 5:
            resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

            pred = model.predict(reshaped, verbose=0)[0]
            confidence = np.max(pred)
            letter = labels[np.argmax(pred)]

            if confidence > 0.7:
                cv2.putText(frame, f'{letter} ({confidence:.2f})', (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
