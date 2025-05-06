import asyncio
import websockets
import json
import base64
import cv2

SERVER_URI = "ws://localhost:8080/ws"

send_queue = None  # не создаём сразу!
event_loop = None

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def resize_image(image, max_width=400):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


async def websocket_sender():
    global send_queue
    if send_queue is None:
        send_queue = asyncio.Queue()  # создаём очередь внутри loop'а

    async with websockets.connect(
            SERVER_URI,
            # max_size=None,  # разрешает принимать любые размеры сообщений
            # max_queue=None,  # не ограничивает очередь входящих сообщений
            # read_limit=2 ** 20 * 10,  # буфер чтения — увеличен
            # write_limit=2 ** 20 * 10  # буфер записи — увеличен
    ) as websocket:
        print("[WS] Connected to server")
        while True:
            frame, name = await send_queue.get()
            resized_image_b64 = resize_image(frame)
            payload = {
                "type": "face_detected",
                "name": name,
                "image": image_to_base64(resized_image_b64)
            }
            if len(json.dumps(payload)) > 950_000:
                await websocket.send(json.dumps("[WARNING] Payload too large, skipping..."))
                return
            else:
                await websocket.send(json.dumps(payload))
            print(f"[WS] Sent: {name}")


def enqueue_to_queue(frame, name):
    global event_loop
    if event_loop and send_queue:
        asyncio.run_coroutine_threadsafe(send_queue.put((frame, name)), event_loop)
