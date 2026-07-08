import cv2
import mediapipe as mp
import numpy as np

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
                break

    #blur efek

    if peace_detected:

        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )
    elif fist_detected:
        blurred = cv2.GaussianBlur(frame, (61, 61), 0)
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 255
        frame = cv2.addWeighted(blurred, 0.7, red_overlay, 0.3, 0)

    cv2.imshow(
        "Peace Blur",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
