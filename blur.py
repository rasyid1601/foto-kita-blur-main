import cv2
import mediapipe as mp
import numpy as np
import time

#detect tangan
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y


def is_peace(landmarks):

    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)

    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    )


def is_fist(landmarks):
    thumb_closed = landmarks[4].x > landmarks[3].x
    index_closed = landmarks[8].y > landmarks[6].y
    middle_closed = landmarks[12].y > landmarks[10].y
    ring_closed = landmarks[16].y > landmarks[14].y
    pinky_closed = landmarks[20].y > landmarks[18].y

    return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed


def draw_fire_effect(frame, landmarks, elapsed):
    height, width = frame.shape[:2]
    points = np.array(
        [(int(point.x * width), int(point.y * height)) for point in landmarks],
        dtype=np.int32
    )
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    hand_width = max(x_max - x_min, 1)
    hand_height = max(y_max - y_min, 1)

    overlay = frame.copy()
    flicker = 0.9 + 0.1 * np.sin(elapsed * 12)
    flame_count = 7

    for index in range(flame_count):
        center_x = x_min + int(hand_width * (index + 0.5) / flame_count)
        base_y = y_max + int(hand_height * 0.18)
        flame_width = max(int(hand_width * 0.24), 8)
        flame_height = max(int(hand_height * (0.55 + 0.18 * np.sin(elapsed * 8 + index))), 12)
        sway = int(np.sin(elapsed * 7 + index * 1.7) * hand_width * 0.12)
        tip_y = y_min - int(hand_height * 0.2) + int(np.sin(elapsed * 9 + index) * hand_height * 0.12)
        points_outer = np.array([
            (center_x - flame_width, base_y),
            (center_x - flame_width // 2, base_y - flame_height // 2),
            (center_x + sway, tip_y),
            (center_x + flame_width, base_y - flame_height // 3),
            (center_x + flame_width // 2, base_y),
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [points_outer], (0, int(105 * flicker), 255))

        points_inner = np.array([
            (center_x - flame_width // 2, base_y),
            (center_x - flame_width // 4, base_y - flame_height // 3),
            (center_x + sway // 2, tip_y + flame_height // 4),
            (center_x + flame_width // 2, base_y),
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [points_inner], (0, 235, 255))

    cv2.ellipse(
        overlay,
        ((x_min + x_max) // 2, y_max),
        (max(hand_width // 2, 10), max(hand_height // 4, 8)),
        0,
        0,
        360,
        (0, 80, 255),
        -1
    )
    return cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

#open camera
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_result = hands.process(rgb)

    peace_detected = False
    fist_detected = False
    fist_landmarks = None

    if hand_result.multi_hand_landmarks:

        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            if is_peace(hand_landmarks.landmark):
                peace_detected = True
                break

            if is_fist(hand_landmarks.landmark):
                fist_detected = True
                fist_landmarks = hand_landmarks.landmark
                break

    #blur efek

    if peace_detected:

        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )
    elif fist_detected:
        frame = draw_fire_effect(frame, fist_landmarks, time.monotonic())

    cv2.imshow(
        "Peace Blur",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
