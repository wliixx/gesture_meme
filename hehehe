import cv2 
import time
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "C:/Users/RedFox3k/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),          # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),     # средний
    (9, 13), (13, 14), (14, 15), (15, 16),   # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),  # мизинец
    (0, 17),                                  # ладонь
]

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=mp_vision.RunningMode.VIDEO
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

TIP_IDS = [4, 8, 12, 16, 20]

def distance(p_a, p_b):
    return math.hypot(p_a.x - p_b.x, p_a.y - p_b.y)

def count_fingers(landmarks, handedness_label):
    fingers = []
    thumb_tip = landmarks[4]
    index_mcp = landmarks[17]
    t_i_d = distance(thumb_tip,index_mcp)
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_size = distance(wrist, middle_mcp)
    THUMB_EXTENDED_RATIO = 0.8
    fingers.append(1 if (t_i_d / hand_size)>THUMB_EXTENDED_RATIO else 0)
        
    for i in range(1,5):
        tip = TIP_IDS[i]
        pip_joint = tip - 2
        fingers.append(1 if landmarks[tip].y < landmarks[pip_joint].y else 0)
    return fingers 

def recognize_gesture(fingers):
    total = sum(fingers)
    known_fingers = {
        (0, 0, 0, 0, 0): "Кулак",
        (1, 1, 1, 1, 1): "Открытая ладонь",
        (0, 1, 0, 0, 0): "Указательный палец",
        (0, 1, 1, 0, 0): "Victory / Мир",
        (0, 1, 0, 0, 1): "Рок (Rock)",
        (1, 1, 0, 0, 1): "Рок (Rock)",
        (1, 0, 0, 0, 0): "Большой палец вверх",
    }
    
    gesture = known_fingers.get(tuple(fingers))
    if gesture:
        return gesture
    return f'Поднято пальцев: {total}'

def draw_hand(frame, landmarks_px):
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx],
(255, 255, 255), 2)
    for x,y in landmarks_px:
        cv2.circle(frame,(x,y),4,(0,0,255), -1)
        

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(0.5)
    
    if not cap.isOpened():
        print('Не удалось открыть камеру')
        return 

    frame_index = 0
    flag = True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Кадр не прочитан, пробуем ещё раз...")
            break 
        if flag:
            print(f"кадр прочитан: ret={ret}, frame is None: {frame is None}")
            flag = False
        frame = cv2.flip(frame,1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_index += 1
        timestamp_ms = frame_index * 33
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            for hand_index,(hand_landmarks, handedness) in  enumerate(zip(result.hand_landmarks, result.handedness)):
                landmarks_px = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
                draw_hand(frame, landmarks_px) 
                label = handedness[0].category_name
                
                fingers = count_fingers(hand_landmarks, label)
                gesture = recognize_gesture(fingers)
                text = f'{label}: {gesture}'
                text_y = 50 + hand_index * 40
                
                cv2.putText(
                    frame,
                    text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0,0,0),
                    3
                )
        cv2.imshow('Gesture Recognizer', frame)
        
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
 
 
if __name__ == "__main__":
    main()