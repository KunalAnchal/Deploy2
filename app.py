from flask import Flask, render_template, Response, request, redirect, url_for
import threading
import time
import cv2
import yaml
from main import process_frame, capture_frames, input_frame_queue, processed_queue, stop_thread, process_frames, display_frames

app = Flask(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Global flag to stop threads
stop_thread = False
threads = []

def start_threads(rtsp_url):
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url,))
    process_thread = threading.Thread(target=process_frames)
    display_thread = threading.Thread(target=display_frames)

    capture_thread.start()
    process_thread.start()
    display_thread.start()

    return [capture_thread, process_thread, display_thread]

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    while not stop_thread:
        if not processed_queue.empty():
            frame = processed_queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global stop_thread, threads
    stop_thread = False
    rtsp_url = config['video_path']
    threads = start_threads(rtsp_url)
    return redirect(url_for('show_feed'))

@app.route('/show_feed')
def show_feed():
    return render_template('result.html')

@app.route('/stop', methods=['POST'])
def stop():
    global stop_thread
    stop_thread = True
    return 'Stopped'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090, debug=True)

