import asyncio
import threading
from threading import Thread
from flask import Flask, request, jsonify
import ws_client
from recognizer import run_recognition_loop

# Flask app
app = Flask(__name__)

# Контрольный флаг и поток
recognition_active = False
recognition_thread = None

# Отдельный loop для WebSocket клиента
ws_loop = asyncio.new_event_loop()
ws_client.event_loop = ws_loop

def start_ws_loop():
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(ws_client.websocket_sender())

# Запуск WebSocket клиента в фоновом потоке
ws_thread = Thread(target=start_ws_loop, daemon=True)
ws_thread.start()

def recognition_worker():
    print("[INFO] Recognition started.")
    run_recognition_loop(callback=ws_client.enqueue_to_queue, control_flag=recognition_event)
    print("[INFO] Recognition stopped.")

recognition_event = threading.Event()

@app.route('/start', methods=['POST'])
def start_recognition():
    global recognition_thread
    if recognition_event.is_set():
        return jsonify({"status": "already running"}), 400

    recognition_event.set()
    recognition_thread = Thread(target=recognition_worker, daemon=True)
    recognition_thread.start()
    return jsonify({"status": "started"}), 200

@app.route('/stop', methods=['POST'])
def stop_recognition():
    if not recognition_event.is_set():
        return jsonify({"status": "not running"}), 400

    recognition_event.clear()
    return jsonify({"status": "stopping"}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"running": recognition_event.is_set()}), 200

if __name__ == '__main__':
    app.run(port=5000)
