import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# --- Функция косинусной близости ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# --- Распознавание ---
def recognize_face(face_emb, known_embs, threshold=0.4):
    best_match = "Unknown"
    best_score = -1

    for name, emb in known_embs.items():
        score = cosine_similarity(face_emb, emb)
        if score > (1 - threshold) and score > best_score:
            best_score = score
            best_match = name

    return f"{best_match}, {best_score}"

# --- Загрузка эталонных эмбеддингов ---
known_faces = {}
for file in os.listdir("embeddings"):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        known_faces[name] = np.load(os.path.join("embeddings", file))

# --- Инициализация модели ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# --- Видеопоток ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        name = recognize_face(face.embedding, known_faces)
        box = face.bbox.astype(int)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
