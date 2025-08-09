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
from libcamera import controls
from enum import Enum


class State(Enum):
    IDLE = 0
    RECORDING = 1


# --- Constants ---
FRAMERATE = 60
FRAME_WIDTH = 1080
FRAME_HEIGHT = 720
JPEG_QUALITY = 85
LORES_WIDTH = 320
LORES_HEIGHT = 240
SAVE_DIR = 'recordings'
MOTION_THRESHOLD = 125
YOLO_CONFIDENCE = 0.5
POST_MOTION_RECORD_SECONDS = 2.0
MIN_RECORD_TIME_SECONDS = 4.0
PRE_RECORD_SECONDS = 2
PRE_RECORD_FRAMES = FRAMERATE * PRE_RECORD_SECONDS
TARGET_CLASSES_TO_IGNORE = [] #['bird', 'airplane', 'insect']
YOLO_FRAME_INTERVAL = 1 
YOLO_VIDEO_ANALYSIS_INTERVAL_SECONDS = 0.1

# [NEW] Add new constants for night mode control
NIGHT_MODE_GAIN_THRESHOLD = 7.0 
DAY_MODE_GAIN_THRESHOLD = 6.0  # Lowered slightly to prevent flickering
NIGHT_EXPOSURE_SECONDS = 2   
NIGHT_ANALOGUE_GAIN = 16.0   

class Camera:
    """
    A redesigned Camera class with a stable, decoupled, and robust pipeline.
    - High-res thread captures frames for streaming and recording.
    - Low-res thread independently handles motion detection.
    - A dedicated queue ensures no frames are dropped during recording.
    - A post-processing step analyzes videos and renames them with descriptive tags.
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
        self.current_recording_path = None 

        # Stream Management
        self.stream_paused_event = threading.Event()
        self.latest_frame_for_stream = None
        self.latest_detections = []
        self.monitoring_active = False 
        self.is_night_mode = False 

        # FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_lock = threading.Lock()

    def start(self):
        """Initializes and starts the camera with auto-exposure controls."""
        print("\n--- Initializing Camera System ---")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
                lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "YUV420"},
                controls={
                    "FrameRate": FRAMERATE,
                    "AeEnable": True,
                    "AwbEnable": True,
                    "ExposureValue": 0.5, 
                    "AeMeteringMode": controls.AeMeteringModeEnum.Matrix,
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.max_exposure_time = self.picam2.camera_controls['ExposureTime'][1]
            print(f"Sensor max exposure time: {self.max_exposure_time} microseconds")
            time.sleep(1.0) 
            print("[SUCCESS] picamera2 started successfully with auto-exposure controls.")
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
                request = self.picam2.capture_request()
                if request is None and self.is_night_mode:
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                with self.fps_lock:
                    if current_time != self.last_frame_time:
                        self.fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time

                main_frame = request.make_array("main")
                request.release()
                
                with self.lock:
                    self.pre_record_buffer.append(main_frame)
                    
                    if self.state == State.RECORDING:
                        self.recording_queue.put(main_frame)
                
                annotated_frame = self._get_annotated_frame(main_frame, draw_status=True)
                with self.lock:
                    self.latest_frame_for_stream = annotated_frame
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                self.running = False
        print("Capture loop has stopped.")


    def _processing_loop(self):
        """Independently captures lo-res frames to detect motion, manage state, and control camera modes."""
        prev_lores_gray = None
        frame_counter = 0

        while self.running:
            try:
                # *** THE FIX IS HERE: This function now contains the corrected logic ***
                self._check_and_update_camera_mode()

                if self.is_night_mode:
                    time.sleep(1) # Check light levels every second
                    continue

                if not self.monitoring_active:
                    if self.latest_detections:
                        self.latest_detections = []
                    time.sleep(0.1)
                    continue

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

                frame_counter += 1
                if frame_counter % (FRAMERATE // YOLO_FRAME_INTERVAL) == 0:
                    main_frame_for_yolo = self.picam2.capture_array("main")
                    results = self.model.predict(main_frame_for_yolo, conf=YOLO_CONFIDENCE, verbose=False)
                    with self.lock:
                        self.latest_detections = results

                if motion_detected:
                    self.last_motion_time = time.time()
                    if self.state == State.IDLE:
                        print("✔️ Motion detected. Starting recording.")
                        self._start_recording()

                elif self.state == State.RECORDING:
                    time_since_start = time.time() - self.recording_start_time
                    time_since_last_motion = time.time() - self.last_motion_time
                    if time_since_start > MIN_RECORD_TIME_SECONDS and time_since_last_motion > POST_MOTION_RECORD_SECONDS:
                        self._stop_recording()

                time.sleep(0.01)

            except Exception as e:
                print(f"Error in processing loop: {e}")
                self.running = False
        print("Processing loop has stopped.")

    def _start_recording(self):
        """Prepares filename, changes state, and starts the recording thread."""
        with self.lock:
            if self.state == State.RECORDING: return
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_recording_path = os.path.join(SAVE_DIR, f"motion_{timestamp}.mp4")

            self.recording_start_time = time.time()
            self.last_motion_time = time.time()
            self.state = State.RECORDING
            self.stream_paused_event.set()

            print(f"Dumping {len(self.pre_record_buffer)} pre-record frames into queue...")
            for frame in list(self.pre_record_buffer):
                self.recording_queue.put(frame)
            
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()

    def _check_and_update_camera_mode(self):
        """ 
        Checks ambient light and switches between day/night modes.
        This function now contains the "peeking" logic to prevent getting stuck.
        """
        if self.is_night_mode:
            # --- PEEKING LOGIC ---
            # 1. Temporarily enable auto-exposure to check the real light level
            self.picam2.set_controls({"AeEnable": True})
            time.sleep(0.5) # Give sensor time to adjust
            
            # 2. Get the new metadata
            metadata = self.picam2.capture_metadata()
            analogue_gain = metadata.get("AnalogueGain", 0)
            
            # 3. Decide whether to switch back to day mode
            if analogue_gain < DAY_MODE_GAIN_THRESHOLD:
                self._enable_day_mode() # It's bright, switch to day mode
            else:
                # It's still dark, revert to manual night mode settings
                self._enable_night_mode(force_revert=True)
        else:
            # --- STANDARD DAYTIME LOGIC ---
            metadata = self.picam2.capture_metadata()
            analogue_gain = metadata.get("AnalogueGain", 0)
            if analogue_gain > NIGHT_MODE_GAIN_THRESHOLD:
                self._enable_night_mode()
    
    def _enable_night_mode(self, force_revert=False):
        """Sets camera controls for long-exposure night viewing."""
        if not force_revert: # Only print the full message on the initial switch
            print(f"🌙 Switching to NIGHT MODE. Analog gain: {self.picam2.capture_metadata()['AnalogueGain']:.2f}")
            if self.monitoring_active:
                print(" Motion monitoring for night mode.")
                #self.stop_monitoring()

        exposure_time_us = int(NIGHT_EXPOSURE_SECONDS * 1_000_000)
        exposure_time_us = min(exposure_time_us, self.max_exposure_time)
        
        controls_to_set = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exposure_time_us,
            "AnalogueGain": NIGHT_ANALOGUE_GAIN,
        }
        self.picam2.set_controls(controls_to_set)
        self.is_night_mode = True
        if not force_revert:
             print(f"   -> Set Exposure: {exposure_time_us/1_000_000:.2f}s, Gain: {NIGHT_ANALOGUE_GAIN}")


    def _enable_day_mode(self):
        """Resets camera controls for standard daytime operation."""
        print("☀️ Switching to DAY MODE.")
        
        controls_to_set = {
            "AeEnable": True, 
            "AwbEnable": True,
            "FrameRate": FRAMERATE
        }
        self.picam2.set_controls(controls_to_set)
        self.is_night_mode = False
        time.sleep(1.0) 

    def _stop_recording(self):
        """Stops recording and triggers the asynchronous video analysis."""
        path_to_analyze = None
        with self.lock:
            if self.state == State.IDLE: return
            print("Stopping recording and finalizing video file...")
            self.state = State.IDLE
            path_to_analyze = self.current_recording_path
        
        if self.recording_thread and self.recording_thread.is_alive():
            print("Waiting for recording thread to finish writing frames...")
            self.recording_thread.join()
            print("Recording thread finished.")
        
        if path_to_analyze:
            print(f"Queueing video for analysis: {os.path.basename(path_to_analyze)}")
            analysis_thread = threading.Thread(
                target=self._analyze_and_rename_video,
                args=(path_to_analyze,),
                daemon=True
            )
            analysis_thread.start()
        
        self.stream_paused_event.clear()
        self.current_recording_path = None

    def _recording_loop(self):
        """Writes frames from the queue to the video file, with correct color space."""
        out_path = self.current_recording_path
        if not out_path:
            print("[Error] Recording loop started without a valid output path.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(out_path, fourcc, float(FRAMERATE), (FRAME_WIDTH, FRAME_HEIGHT))
        
        print(f"Recording to {out_path}...")
        while self.state == State.RECORDING or not self.recording_queue.empty():
            try:
                frame_bgr = self.recording_queue.get(timeout=1)
                annotated_frame_bgr = self._get_annotated_frame(frame_bgr, draw_status=False)
                
                frame_to_write_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                self.video_writer.write(frame_to_write_rgb)
                
            except queue.Empty:
                break
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"Video saved: {out_path}")

    def _analyze_and_rename_video(self, video_path):
        """Opens a completed video, runs YOLO detection, and renames the file with tags."""
        if not video_path or not os.path.exists(video_path):
            print(f"[Analysis] Video path not found or invalid: {video_path}")
            return

        print(f"[Analysis] Starting analysis for {os.path.basename(video_path)}...")
        detected_objects = set()
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Analysis] Error: Could not open video file {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * YOLO_VIDEO_ANALYSIS_INTERVAL_SECONDS)
            if frame_interval < 1: frame_interval = 1 

            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    results = self.model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            class_name = self.model.names[int(box.cls)]
                            if class_name not in TARGET_CLASSES_TO_IGNORE:
                                detected_objects.add(class_name)
                frame_number += 1
            
            cap.release()

            if detected_objects:
                tags = "_".join(sorted(list(detected_objects)))
                path_without_ext, ext = os.path.splitext(video_path)
                new_path = f"{path_without_ext}_{tags}{ext}"
                
                os.rename(video_path, new_path)
                print(f"[Analysis] Success! Renamed video to: {os.path.basename(new_path)}")
            else:
                print("[Analysis] No relevant objects detected to add tags.")

        except Exception as e:
            print(f"[Analysis] An error occurred during video analysis: {e}")

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
            if self.is_night_mode:
                mode_text, mode_color = "MODE: NIGHT", (255, 255, 0) # Cyan for night
            else:
                mode_text, mode_color = "MODE: DAY", (0, 255, 255) # Yellow for day
            cv2.putText(annotated_frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            
            status_text = f"Monitoring: {'ON' if self.monitoring_active else 'OFF'}"
            color = (0, 255, 0) if self.monitoring_active else (100, 100, 100)
            if self.state == State.RECORDING:
                status_text, color = "Status: RECORDING", (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        with self.fps_lock:
            if self.is_night_mode:
                info_text = f"Exposure: {NIGHT_EXPOSURE_SECONDS}s"
            else:
                info_text = f"{self.fps:.1f} FPS"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        combined_text = f"{timestamp} | {info_text}"
        cv2.putText(annotated_frame, combined_text, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def get_jpeg_frame(self):
        """Encodes the latest streamable frame into a JPEG with correct color space."""
        with self.lock:
            if self.latest_frame_for_stream is None:
                frame_bgr = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_bgr, "Initializing...", (int(FRAME_WIDTH/2)-100, int(FRAME_HEIGHT/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame_bgr = self.latest_frame_for_stream
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        ret, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
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

    def start_monitoring(self):
        """Enables motion detection and automatic recording."""
        print("Monitoring has been ENABLED by user.")
        self.monitoring_active = True

    def stop_monitoring(self):
        """Disables motion detection and automatic recording."""
        print("Monitoring has been DISABLED by user.")
        self.monitoring_active = False
        if self.state == State.RECORDING:
            self._stop_recording()


#------------------------------------------
# Flask App Section
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
            print("A file was changed during list generation. Refreshing list.")
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
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    placeholder[:] = (68, 68, 68) # BGR
    text = "RECORDING IN PROGRESS"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (FRAME_WIDTH - text_size[0]) // 2
    text_y = (FRAME_HEIGHT + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    ret, placeholder_buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    placeholder_bytes = placeholder_buffer.tobytes() if ret else b''

    while True:
        if cam.stream_paused_event.is_set():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        frame_bytes = cam.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1 / (FRAMERATE * 2))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_stream(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_recording(filename):
    try:
        file_path = os.path.join(SAVE_DIR, filename)
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

    return redirect(url_for('index'))

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    camera.start_monitoring()
    return redirect(url_for('index'))

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
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
