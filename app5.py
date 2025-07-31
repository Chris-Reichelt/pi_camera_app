import cv2
import time
import os
import threading
import collections
import queue
import numpy as np
from flask import Flask, Response, send_from_directory, redirect, url_for
from ultralytics import YOLO

import sys
from unittest.mock import Mock
sys.modules['pykms'] = Mock()

from picamera2.picamera2 import Picamera2
import libcamera
from enum import Enum

class State(Enum):
    IDLE = 0
    RECORDING = 1


FRAMERATE = 60
FRAME_WIDTH = 1080 #1920 #1080
FRAME_HEIGHT = 720 #1080 #720
JPEG_QUALITY = 85
LORES_WIDTH = 320
LORES_HEIGHT = 240
SAVE_DIR = 'recordings'
MOTION_THRESHOLD = 20
YOLO_CONFIDENCE = 0.51
YOLO_FRAME_INTERVAL = 2
POST_MOTION_RECORD_SECONDS = 5.0
MIN_RECORD_TIME_SECONDS = 10.0 
PRE_RECORD_SECONDS = 5
PRE_RECORD_FRAMES = FRAMERATE * PRE_RECORD_SECONDS
TARGET_CLASSES_TO_IGNORE = []#['bird', 'airplane','insect']

class Camera:
    """
    A redesigned Camera class with a stable, decoupled, and robust pipeline.
    - High-res thread captures frames for streaming and recording.
    - Low-res thread independently handles motion detection.
    - A dedicated queue ensures no frames are dropped during recording.
    """

    def __init__(self):
        self.picam2 = None
        self.model = YOLO('yolov8n.pt')

        self.state = State.IDLE
        self.lock = threading.Lock()
        self.running = False
        
        # Threading
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.recording_thread = None

        # Data Pipelines
        self.pre_record_buffer = collections.deque(maxlen=PRE_RECORD_FRAMES)
        self.recording_queue = queue.Queue()

        # State and Timers
        self.last_motion_time = 0
        self.recording_start_time = 0
        self.video_writer = None

        # Stream Management
        self.stream_paused_event = threading.Event()
        self.latest_frame_for_stream = None
        self.latest_detections = []
        self.monitoring_active = False # Start with monitoring ON by default

    def start(self):
        """Initializes and starts the camera with a stable configuration."""
        print("\n--- Initializing Camera System ---")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
                lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "YUV420"},
                controls={"FrameRate": FRAMERATE}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            print("[SUCCESS] picamera2 started successfully.")
        except Exception as e:
            print(f"[FAIL] Failed to initialize picamera2: {e}")
            self.running = False
            return

        self.running = True
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.capture_thread.start()
        self.processing_thread.start()
        print("Camera capture and processing threads started.")

    def _capture_loop(self):
        """Captures hi-res frames, manages the pre-record buffer, and feeds the recording queue."""
        while self.running:
            try:
                main_frame = self.picam2.capture_array("main")
                
                with self.lock:
                    # Always keep the pre-record buffer updated.
                    self.pre_record_buffer.append(main_frame)
                    
                    # If we are recording, also put the frame into the dedicated recording queue.
                    if self.state == State.RECORDING:
                        self.recording_queue.put(main_frame)
                
                # Annotate and update the frame for the live stream.
                annotated_frame = self._get_annotated_frame(main_frame, draw_status=True)
                with self.lock:
                    self.latest_frame_for_stream = annotated_frame
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                self.running = False
        print("Capture loop has stopped.")

    def _processing_loop(self):
        """Independently captures lo-res frames to detect motion and manage state."""
        prev_lores_gray = None
        frame_count = 0
        while self.running:
            try:
                # [NEW] If monitoring is off, skip all processing.
                if not self.monitoring_active:
                    # Clear any old detections so they don't stay on screen.
                    if self.latest_detections:
                        self.latest_detections = []
                    time.sleep(0.1)
                    continue

                # --- The rest of the processing logic is now nested under the active check ---
                lores_frame = self.picam2.capture_array("lores")
                
                lores_gray = lores_frame[0:LORES_HEIGHT, 0:LORES_WIDTH]
                lores_gray = cv2.GaussianBlur(lores_gray, (15, 15), 0)
                if prev_lores_gray is None:
                    prev_lores_gray = lores_gray
                    continue
                
                frame_delta = cv2.absdiff(prev_lores_gray, lores_gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_detected = cv2.countNonZero(thresh) > MOTION_THRESHOLD
                prev_lores_gray = lores_gray

                if motion_detected:
                    self.last_motion_time = time.time()
                    if self.state == State.IDLE:
                        frame_count += 1
                        if frame_count % YOLO_FRAME_INTERVAL == 0:
                            with self.lock:
                                yolo_frame = self.pre_record_buffer[-1].copy() if self.pre_record_buffer else None
                            
                            if yolo_frame is not None:
                                results = self.model(yolo_frame, conf=YOLO_CONFIDENCE, verbose=False)
                                self.latest_detections = results
                                detected_classes = [self.model.names[int(box.cls)] for r in results for box in r.boxes]

                                if detected_classes and not any(cls in TARGET_CLASSES_TO_IGNORE for cls in detected_classes):
                                    print(f"✔️ Validated motion: {detected_classes}. Starting recording.")
                                    self._start_recording()
                
                elif self.state == State.RECORDING:
                    time_since_start = time.time() - self.recording_start_time
                    time_since_last_motion = time.time() - self.last_motion_time
                    if time_since_start > MIN_RECORD_TIME_SECONDS and time_since_last_motion > POST_MOTION_RECORD_SECONDS:
                        self._stop_recording()

                time.sleep(0.01)

            except Exception as e:
                print(f"Error in motion detection loop: {e}")
                self.running = False
        print("Motion detection loop has stopped.")


    def _start_recording(self):
        """Changes state and starts the recording thread."""
        with self.lock:
            if self.state == State.RECORDING: return
            
            self.recording_start_time = time.time()
            self.last_motion_time = time.time()
            self.state = State.RECORDING
            self.stream_paused_event.set()

            print(f"Dumping {len(self.pre_record_buffer)} pre-record frames into queue...")
            for frame in list(self.pre_record_buffer):
                self.recording_queue.put(frame)
            
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()

    def _stop_recording(self):
        """[FIX] This method is now included. It stops the recording and waits for the thread to finish."""
        with self.lock:
            if self.state == State.IDLE: return
            print("Stopping recording and finalizing video file...")
            self.state = State.IDLE
        
        if self.recording_thread and self.recording_thread.is_alive():
            print("Waiting for recording thread to finish writing frames...")
            self.recording_thread.join()
            print("Recording thread finished.")
        
        self.stream_paused_event.clear()

    def _recording_loop(self):
        """A simple, robust consumer that writes frames from the queue to a video file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"motion_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(out_path, fourcc, float(FRAMERATE), (FRAME_WIDTH, FRAME_HEIGHT))
        
        print(f"Recording to {out_path}...")
        while self.state == State.RECORDING or not self.recording_queue.empty():
            try:
                # Get the BGR frame from the queue
                frame_bgr = self.recording_queue.get(timeout=1)
                
                # Annotate the BGR frame
                annotated_frame_bgr = self._get_annotated_frame(frame_bgr, draw_status=False)
                
                # [FIX] Convert the final annotated frame to RGB for the video encoder.
                frame_to_write_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                
                self.video_writer.write(frame_to_write_rgb)

            except queue.Empty:
                # This can happen if the state changes right as the queue empties.
                break
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"Video saved: {out_path}")

    def _get_annotated_frame(self, frame, draw_status=False):
        """Draws annotations on a copy of the frame."""
        annotated_frame = frame.copy()
        
        if self.latest_detections:
            for r in self.latest_detections:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf, cls_id = float(box.conf), int(box.cls)
                    label = f"{self.model.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if draw_status:
            status_text, color = (f"Monitoring Status:{self.monitoring_active}", (0, 255, 0))
            if self.state == State.RECORDING:
                status_text, color = "Status: RECORDING", (0, 0, 255)
            # You could add another state/flag for "motion detected" if desired
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(annotated_frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated_frame

    def get_jpeg_frame(self):
        """Encodes the latest streamable frame into a JPEG."""
        with self.lock:
            if self.latest_frame_for_stream is None:
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "Initializing...", (int(FRAME_WIDTH/2)-100, int(FRAME_HEIGHT/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = self.latest_frame_for_stream
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buffer.tobytes() if ret else None

    def stop(self):
        """Stops all threads and releases the camera."""
        print("--- Shutting Down Camera System ---")
        self.running = False
        
        if self.capture_thread.is_alive(): self.capture_thread.join(timeout=2)
        if self.processing_thread.is_alive(): self.processing_thread.join(timeout=2)
        
        if self.state == State.RECORDING:
             self._stop_recording()

        if self.picam2:
            self.picam2.stop()
            print("picamera2 stopped.")
        print("Camera system stopped.")

    def get_jpeg_frame(self):
        """Encodes the latest streamable frame into a JPEG, fixing the color space."""
        with self.lock:
            if self.latest_frame_for_stream is None:
                frame_bgr = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, "Initializing...", (int(FRAME_WIDTH/2)-100, int(FRAME_HEIGHT/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame_bgr = self.latest_frame_for_stream
        
        # [FIX] Convert from OpenCV's BGR to standard RGB for web display.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        ret, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buffer.tobytes() if ret else None

    def stop(self):
        """Stops all threads and releases the camera."""
        print("--- Shutting Down Camera System ---")
        self.running = False
        
        # Wait for threads to finish their current loop
        if self.capture_thread.is_alive(): self.capture_thread.join(timeout=2)
        if self.processing_thread.is_alive(): self.processing_thread.join(timeout=2)
        
        # The stop_recording logic handles joining the recording thread
        if self.state == State.RECORDING:
             self._stop_recording()

        if self.picam2:
            self.picam2.stop()
            print("picamera2 stopped.")
        print("Camera system stopped.")

    def start_monitoring(self):
        """Enables motion detection and automatic recording."""
        print("Monitoring has been ENABLED by user.")
        self.monitoring_active = True

    def stop_monitoring(self):
        """Disables motion detection and automatic recording."""
        print("Monitoring has been DISABLED by user.")
        self.monitoring_active = False
        # When monitoring is turned off, also stop any recording that might be in progress.
        self._stop_recording()

#------------------------------------------
app = Flask(__name__)
camera = Camera()

@app.route('/')
def index():
    """Serves the main HTML page and lists recordings."""
    recordings = []
    if os.path.exists(SAVE_DIR):
        all_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp4')]
        try:
            all_files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)), reverse=True)
        except FileNotFoundError:
            print("A recording was deleted while generating the list. Refreshing.")
            all_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.mp4')]
            all_files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)), reverse=True)
        recordings = all_files

    recordings_html = '<h2>No recordings yet.</h2>'
    if recordings:
        recordings_html = '<h2>Recordings</h2><ul id="recordings-list">'
        for rec in recordings:
            recordings_html += f'''
                <li>
                    <a href="/recordings/{rec}" target="_blank">{rec}</a>
                    <form action="/delete/{rec}" method="POST" style="display:inline;">
                        <button type="submit">Delete</button>
                    </form>
                </li>
            '''
        recordings_html += '</ul>'

    # [UPDATED] Logic now checks the 'monitoring_active' flag
    if camera.monitoring_active:
        controls_html = f'''
            <form action="/stop_monitoring" method="POST">
                <button type="submit" class="control-button stop">Stop Monitoring</button>
            </form>
            <p>Status: Monitoring for motion...</p>
        '''
    else:
        controls_html = '''
            <form action="/start_monitoring" method="POST">
                <button type="submit" class="control-button start">Start Monitoring</button>
            </form>
            <p>Status: Monitoring is OFF.</p>
        '''

    # The rest of the HTML structure remains the same as before.
    return f'''
    <html>
        <head>
            <title>Raspberry Pi Motion Camera</title>
            <style>
                body {{ font-family: sans-serif; background-color: #222; color: #eee; margin: 0; padding: 0; }}
                h1 {{ text-align: center; padding: 20px; background-color: #000; }}
                .stream-container {{ max-width: 90%; margin: 20px auto; border: 3px solid #444; border-radius: 8px; overflow: hidden; }}
                img {{ display: block; width: 100%; height: auto; }}
                
                .controls-container {{ text-align: center; margin: 20px; }}
                .controls-container p {{ color: #aaa; }}
                .control-button {{ font-size: 1.2em; padding: 10px 20px; border-radius: 8px; border: none; color: white; cursor: pointer; transition: background-color 0.2s; }}
                .control-button.start {{ background-color: #28a745; }}
                .control-button.start:hover {{ background-color: #218838; }}
                .control-button.stop {{ background-color: #dc3545; }}
                .control-button.stop:hover {{ background-color: #c82333; }}

                .recordings-container {{ max-width: 90%; margin: 20px auto; padding: 10px 20px; background-color: #333; border: 1px solid #444; border-radius: 8px; }}
                .recordings-container h2 {{ color: #eee; border-bottom: 1px solid #555; padding-bottom: 10px; }}
                .recordings-container ul {{ list-style: none; padding: 0; }}
                .recordings-container li {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #444; }}
                .recordings-container li a {{ color: #3498db; text-decoration: none; font-size: 1.1em; }}
                .recordings-container li a:hover {{ text-decoration: underline; }}
                .recordings-container button {{ background-color: #e74c3c; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; }}
                .recordings-container button:hover {{ background-color: #c0392b; }}
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Motion Camera</h1>
            <div class="stream-container">
                <img src="/video_feed">
            </div>
            
            <div class="controls-container">
                {controls_html}
            </div>

            <div class="recordings-container">
                {recordings_html}
            </div>
        </body>
    </html>
    '''

def generate_frames_for_stream(cam):
    # Create a static "Recording..." placeholder frame once
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    placeholder[:] = (68, 68, 68) # A dark gray background (BGR format)
    text = "RECORDING IN PROGRESS"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (FRAME_WIDTH - text_size[0]) // 2
    text_y = (FRAME_HEIGHT + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ret, placeholder_buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        print("Error: Could not encode the placeholder image!")
        # Create a tiny fallback placeholder to avoid crashing the stream
        placeholder_bytes = b''
    else:
        placeholder_bytes = placeholder_buffer.tobytes()


    while True:
        # Check if the stream should be paused
        if cam.stream_paused_event.is_set():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
            time.sleep(0.5) # Send the placeholder every half second
            continue

        # If not paused, stream normally
        frame_bytes = cam.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # A small sleep to prevent the web stream from overwhelming the client
        time.sleep(1 / (FRAMERATE * 2))



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_stream(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    """Serves a recorded video file from the recordings directory."""
    return send_from_directory(SAVE_DIR, filename)

# New route to handle deleting video files
@app.route('/delete/<path:filename>', methods=['POST'])
def delete_recording(filename):
    """Deletes a recorded video file."""
    try:
        # Security: ensure the filename is safe and only targets the recordings directory
        file_path = os.path.join(SAVE_DIR, filename)
        # Check that the resolved path is still within the SAVE_DIR
        if os.path.commonprefix((os.path.realpath(file_path), os.path.realpath(SAVE_DIR))) != os.path.realpath(SAVE_DIR):
             return "Invalid filename.", 400

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted recording: {filename}")
        else:
            print(f"Attempted to delete non-existent file: {filename}")
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")
        return "Error deleting file.", 500

    # Redirect back to the main page to see the updated list
    return redirect(url_for('index'))

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Endpoint to enable monitoring."""
    camera.start_monitoring()
    return redirect(url_for('index'))

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Endpoint to disable monitoring."""
    camera.stop_monitoring()
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        camera.start()
        if camera.running:
            print("\n--- Starting Flask Web Server ---")
            app.run(host='0.0.0.0', port=5000, threaded=True)
        else:
            print("\n--- Application startup failed: Could not initialize camera. ---")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down.")
    finally:
        camera.stop()
