import cv2
import numpy as np
import os
import time
from insightface.app import FaceAnalysis
from numpy.linalg import norm


def load_embeddings(path="embeddings"):
    known_faces = {}
    for file in os.listdir(path):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            known_faces[name] = np.load(os.path.join(path, file))
    return known_faces


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def recognize_face(face_emb, known_embs, threshold=0.4):
    best_match = "Unknown"
    best_score = -1
    for name, emb in known_embs.items():
        score = cosine_similarity(face_emb, emb)
        if score > (1 - threshold) and score > best_score:
            best_score = score
            best_match = name
    return best_match


def run_recognition_loop(callback=None, control_flag=None, threshold=0.4, cooldown_seconds=1):
    """
    Основной цикл распознавания.
    Показывает изображение с рамкой и вызывает callback при появлении нового лица.

    :param callback: функция вида callback(frame_with_box, name)
    :param cooldown_seconds: минимальное время (в секундах) между повторами для одного и того же лица
    """
    known_faces = load_embeddings()
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    cap = cv2.VideoCapture(0)

    # Последнее время вызова callback для каждого имени
    last_seen = {}

    while control_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        current_time = time.time()

        for face in faces:
            name = recognize_face(face.embedding, known_faces, threshold)
            box = face.bbox.astype(int)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Проверка на "новизну" обнаружения
            last_time = last_seen.get(name, 0)
            if current_time - last_time >= cooldown_seconds:
                if name != "Unknown":
                    callback(frame.copy(), name)
                last_seen[name] = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
