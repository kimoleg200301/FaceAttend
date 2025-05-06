import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Инициализация модели
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Пути к фото одного и того же человека
image_paths = [
    "faces/32_1.jpg",
    "faces/32_2.jpg",
    "faces/32_3.jpg"
]

embeddings = []

# Обработка всех фото
for path in image_paths:
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        continue
    embeddings.append(faces[0].embedding)

# Усреднение эмбеддингов
average_embedding = np.mean(embeddings, axis=0)

# Сохраняем в файл
np.save("embeddings/32.npy", average_embedding)
