import cv2
import time
import os
import threading
import collections
import queue
import numpy as np
import io, zipfile
from flask import Flask, Response, send_from_directory, redirect, url_for, request, abort, send_file, flash, get_flashed_messages

# --- Mock pykms for non-Pi environments ---
import sys
from unittest.mock import Mock
sys.modules['pykms'] = Mock()

from picamera2.picamera2 import Picamera2
import libcamera
from libcamera import controls
from enum import Enum
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
try:
    from astral import LocationInfo
    from astral.sun import sun
    HAVE_ASTRAL = True
except ImportError:
    HAVE_ASTRAL = False

class State(Enum):
    IDLE = 0
    RECORDING = 1

# --- Constants ---
# --- Simplified and Retuned ---
FRAMERATE = 30
FRAME_WIDTH = 1600
FRAME_HEIGHT = 900
JPEG_QUALITY = 95
LORES_WIDTH = 640
LORES_HEIGHT = 480
SAVE_DIR = 'recordings'

# --- NEW: Motion Tracking & Recording Logic ---
MIN_TRACK_AGE = 5             # Object must be tracked for this many frames to trigger recording.
POST_MOTION_RECORD_SECONDS = 8.0 # Increased significantly to avoid early cut-offs.
MIN_RECORD_TIME_SECONDS = 5.0    # Minimum length of any saved video.
MAX_TRACK_DISTANCE = 90          # Max pixels a tracked object can move between lo-res frames.

# --- Day Mode Motion Tuning ---
DAY_MIN_AREA = 20           # Increased to ignore insects.
DAY_MAX_AREA = 8000         # Ignores massive changes like cloud shadows.
DAY_MOTION_THRESHOLD = 20   # How sensitive the background subtraction is.

# --- Night Mode Motion Tuning ---
NIGHT_BRIGHTNESS_THRESHOLD = 35 # How bright a spot must be to be considered.
NIGHT_MIN_AREA = 3              # Smallest pixel area for a night object.
NIGHT_MAX_AREA = 500            # Largest pixel area for a night object.

# --- Camera & Time Settings (largely unchanged) ---
PRE_RECORD_SECONDS = 2
PRE_RECORD_FRAMES = FRAMERATE * PRE_RECORD_SECONDS
SUN_TRANSITION_MINUTES = 50
NIGHT_EXPOSURE_SECONDS = 1
NIGHT_ANALOGUE_GAIN = 14.0
LATITUDE = 35.9211
LONGITUDE = -80.5221
TIMEZONE = "America/New_York"

class Camera:
    """
    A radically simplified Camera class that uses robust motion tracking instead of YOLO.
    - It focuses on persistent motion to filter out false positives like insects.
    - Recording logic is improved to capture the entire event without premature cut-offs.
    - Day/Night modes are preserved for optimal sensor performance.
    """
    def __init__(self):
        self.picam2 = None
        # --- YOLO has been completely removed ---
        self.state = State.IDLE
        self.lock = threading.Lock()
        self.running = False
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.recording_thread = None

        self.pre_record_buffer = collections.deque(maxlen=PRE_RECORD_FRAMES)
        self.recording_queue = queue.Queue()

        self.last_motion_time = 0
        self.recording_start_time = 0
        self.current_recording_path = None
        self._last_mode_switch_ts = 0.0
        self.prev_lores_gray = None
        
        # --- NEW: Simple Object Tracker State ---
        self.next_track_id = 0
        self.active_tracks = {} # {id: {'pos': (x,y), 'age': frames, 'last_seen': frame_count}}
        self.frame_count = 0

        self.stream_paused_event = threading.Event()
        self.latest_frame_for_stream = None
        self.monitoring_active = False 
        self.is_night_mode = False 
        self.latest_lores_gray = None
        
        self._sunrise_today = None
        self._sunset_today = None
        self._sun_refresh_after = None

        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_lock = threading.Lock()

    # --- start(), _refresh_sun_times(), and _decide_initial_mode_by_time() are largely unchanged ---
    # (Minor formatting and print statement changes for clarity)
    def start(self):
        print("\n--- Initializing Camera System ---")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "YUV420"},
                controls={"FrameRate": FRAMERATE, "AeEnable": True, "AwbEnable": True},
                buffer_count=2
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.max_exposure_time = self.picam2.camera_controls['ExposureTime'][1]
            time.sleep(1.0) 
            self._decide_initial_mode_by_time()
            print("[SUCCESS] Picamera2 started successfully.")
        except Exception as e:
            print(f"[FAIL] Failed to initialize picamera2: {e}")
            self.running = False
            return

        self.running = True
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.capture_thread.start()
        self.processing_thread.start()
        print("Camera capture and processing threads started.")
        self._refresh_sun_times()
        print(f"[Sun] Sunrise today: {self._sunrise_today}, Sunset: {self._sunset_today}")

    def _refresh_sun_times(self):
        """Compute today's sunrise/sunset in local tz; refresh again just after midnight."""
        tz = ZoneInfo(TIMEZONE)
        today = date.today()
        if HAVE_ASTRAL:
            loc = LocationInfo(latitude=LATITUDE, longitude=LONGITUDE, timezone=TIMEZONE)
            s = sun(loc.observer, date=today, tzinfo=tz)
            self._sunrise_today = s["sunrise"]
            self._sunset_today = s["sunset"]
        else:
            now = datetime.now(tz)
            self._sunrise_today = now.replace(hour=6, minute=30, second=0, microsecond=0)
            self._sunset_today  = now.replace(hour=18, minute=30, second=0, microsecond=0)
        tomorrow = datetime.now(tz).date() + timedelta(days=1)
        self._sun_refresh_after = datetime.combine(tomorrow, datetime.min.time(), tz) + timedelta(minutes=1)

    def _decide_initial_mode_by_time(self):
        """One-shot day/night classification at startup using sunrise/sunset times."""
        try:
            if not self._sunrise_today: self._refresh_sun_times()
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
            day_starts = self._sunrise_today - timedelta(minutes=SUN_TRANSITION_MINUTES)
            night_starts = self._sunset_today + timedelta(minutes=SUN_TRANSITION_MINUTES)
            if day_starts <= now < night_starts:
                print(f"[InitMode] It's daytime ({now.strftime('%H:%M')}). Starting in DAY mode.")
                self._enable_day_mode()
            else:
                print(f"[InitMode] It's nighttime ({now.strftime('%H:%M')}). Starting in NIGHT mode.")
                self._enable_night_mode()
            self._last_mode_switch_ts = time.time()
        except Exception as e:
            print(f"[InitMode] Fallback to DAY mode due to error: {e}")
            self._enable_day_mode()

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
                    if current_time > self.last_frame_time:
                        self.fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time

                main_frame = request.make_array("main")
                lores_frame = request.make_array("lores")
                lores_gray = lores_frame[0:LORES_HEIGHT, 0:LORES_WIDTH]

                with self.lock:
                    self.pre_record_buffer.append(main_frame)
                    if self.state == State.RECORDING:
                        self.recording_queue.put(main_frame)
                    self.latest_lores_gray = lores_gray.copy()
                request.release()
                
                annotated_frame = self._get_annotated_frame(main_frame, draw_status=True)
                with self.lock:
                    self.latest_frame_for_stream = annotated_frame
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                self.running = False
        print("Capture loop has stopped.")

    def _processing_loop(self):
        """Independently analyzes lo-res frames to find and track moving objects."""
        while self.running:
            try:
                if self.state == State.IDLE: self._update_mode_by_time()

                if not self.monitoring_active:
                    time.sleep(0.1)
                    continue

                COOLDOWN_SECONDS = 5.0
                if time.time() - self._last_mode_switch_ts < COOLDOWN_SECONDS:
                    self.active_tracks.clear()  # Ensure tracker is clean during cooldown
                    self.prev_lores_gray = None # Reset background model for day mode
                    time.sleep(0.5)             # Wait before checking again
                    continue

                # --- NEW: Unified Motion Detection & Tracking ---
                with self.lock:
                    lores_gray = None if self.latest_lores_gray is None else self.latest_lores_gray.copy()
                
                if lores_gray is not None:
                    # 1. Find potential motion contours based on current mode
                    contours = self._detect_potential_motion(lores_gray)
                    
                    # 2. Update our object tracker with these contours
                    self._update_tracker(contours)

                # 3. Check if any tracked object is "confirmed" (i.e., old enough)
                confirmed_tracks = [t for t in self.active_tracks.values() if t['age'] >= MIN_TRACK_AGE]
                
                if confirmed_tracks:
                    self.last_motion_time = time.time()
                    if self.state == State.IDLE:
                        print(f"✔️ Confirmed track (age >= {MIN_TRACK_AGE}). Starting recording.")
                        self._start_recording()
                
                elif self.state == State.RECORDING:
                    time_since_start = time.time() - self.recording_start_time
                    time_since_last_motion = time.time() - self.last_motion_time
                    if time_since_start > MIN_RECORD_TIME_SECONDS and time_since_last_motion > POST_MOTION_RECORD_SECONDS:
                        self._stop_recording()

                time.sleep(0.02) # Give a small sleep to prevent busy-waiting

            except Exception as e:
                print(f"Error in processing loop: {e}")
                self.running = False
        print("Processing loop has stopped.")
        
    def _detect_potential_motion(self, lores_gray):
        """Performs image processing to find contours of moving objects."""
        contours = []
        if self.is_night_mode:
            # For night, find bright spots
            _, thresh = cv2.threshold(lores_gray, NIGHT_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area, max_area = NIGHT_MIN_AREA, NIGHT_MAX_AREA
        else:
            # For day, use background subtraction
            lores_gray_blurred = cv2.GaussianBlur(lores_gray, (9, 9), 0)
            if self.prev_lores_gray is None:
                self.prev_lores_gray = lores_gray_blurred
                return []
            
            frame_delta = cv2.absdiff(self.prev_lores_gray, lores_gray_blurred)
            self.prev_lores_gray = lores_gray_blurred
            
            thresh = cv2.threshold(frame_delta, DAY_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area, max_area = DAY_MIN_AREA, DAY_MAX_AREA

        # Filter contours by area
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        return valid_contours

    def _update_tracker(self, contours):
        """A simple centroid tracker to follow objects frame-to-frame."""
        self.frame_count += 1
        
        # Get centroids of current contours
        current_centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_centroids.append({'pos': (cx, cy), 'matched': False})
        
        # Match with active tracks
        for track_id, track_data in self.active_tracks.items():
            best_match_dist = MAX_TRACK_DISTANCE
            best_centroid = None
            
            for centroid in current_centroids:
                if not centroid['matched']:
                    dist = np.linalg.norm(np.array(track_data['pos']) - np.array(centroid['pos']))
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_centroid = centroid
            
            if best_centroid:
                track_data['pos'] = best_centroid['pos']
                track_data['age'] += 1
                track_data['last_seen'] = self.frame_count
                best_centroid['matched'] = True

        # Add new unmatched centroids as new tracks
        for centroid in current_centroids:
            if not centroid['matched']:
                self.active_tracks[self.next_track_id] = {
                    'pos': centroid['pos'], 'age': 1, 'last_seen': self.frame_count
                }
                self.next_track_id += 1
        
        # Prune old tracks that haven't been seen
        stale_ids = [tid for tid, t in self.active_tracks.items() if self.frame_count - t['last_seen'] > 3]
        for tid in stale_ids:
            del self.active_tracks[tid]
            
    def _update_mode_by_time(self):
        """Switches between day and night modes based on sunrise/sunset."""
        now_ts = time.time()
        if now_ts - self._last_mode_switch_ts < 60.0: return

        try:
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
            if self._sun_refresh_after and now >= self._sun_refresh_after:
                self._refresh_sun_times()

            day_starts = self._sunrise_today - timedelta(minutes=SUN_TRANSITION_MINUTES)
            night_starts = self._sunset_today + timedelta(minutes=SUN_TRANSITION_MINUTES)
            is_currently_daytime = day_starts <= now < night_starts

            mode_changed = False
            if is_currently_daytime and self.is_night_mode:
                print(f"[ModeCheck] Time ({now.strftime('%H:%M')}) is past day threshold. Switching to DAY.")
                mode_changed = True
                self._enable_day_mode()
            elif not is_currently_daytime and not self.is_night_mode:
                print(f"[ModeCheck] Time ({now.strftime('%H:%M')}) is past night threshold. Switching to NIGHT.")
                mode_changed = True
                self._enable_night_mode()

            if mode_changed:
                print("Mode switched. Resetting motion background and tracker.")
                self.prev_lores_gray = None
                self.active_tracks.clear()
                self._last_mode_switch_ts = now_ts
                time.sleep(2.0)

        except Exception as e:
            print(f"[ModeCheck] Error during time-based mode check: {e}")

    # --- _enable_night_mode() and _enable_day_mode() are largely unchanged ---
    def _enable_night_mode(self):
        """Lock long exposure / high gain for night viewing."""
        if not hasattr(self, 'max_exposure_time'):
            self.max_exposure_time = self.picam2.camera_controls['ExposureTime'][1]
        
        print(f"🌙 Switching to NIGHT MODE.")
        exposure_time_us = min(int(NIGHT_EXPOSURE_SECONDS * 1_000_000), int(self.max_exposure_time))
        self.picam2.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": exposure_time_us,
            "AnalogueGain": float(NIGHT_ANALOGUE_GAIN),
            "FrameRate": 1 / NIGHT_EXPOSURE_SECONDS,
        })
        self.is_night_mode = True
        print(f"    -> Set Exposure: {exposure_time_us / 1_000_000:.2f}s, Gain: {NIGHT_ANALOGUE_GAIN}")

    def _enable_day_mode(self):
        """Return to normal auto-exposure/framerate."""
        print("☀️ Switching to DAY MODE.")
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True, "FrameRate": FRAMERATE})
        self.is_night_mode = False
        time.sleep(1.0) # let AE converge

    # --- _start_recording() and _stop_recording() are simplified ---
    def _start_recording(self):
        """Prepares filename and starts the recording thread."""
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

    def _stop_recording(self):
        """Signals the recording thread to stop and waits for it to finish."""
        with self.lock:
            if self.state == State.IDLE: return
            print("Stopping recording...")
            self.state = State.IDLE
            self.current_recording_path = None

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=10.0) 
            if self.recording_thread.is_alive():
                print("Error: Recording thread is unresponsive.")
            else:
                print("Recording thread finished.")
        
        self.stream_paused_event.clear()

    # --- _recording_loop() no longer calls the analyzer ---
    def _recording_loop(self):
        """Writes frames from the queue to the video file."""
        out_path = self.current_recording_path
        if not out_path: return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, float(FRAMERATE), (FRAME_WIDTH, FRAME_HEIGHT))
        
        print(f"Recording to {out_path}...")
        while self.state == State.RECORDING or not self.recording_queue.empty():
            try:
                frame_rgb = self.recording_queue.get(timeout=1)
                annotated_frame_rgb = self._get_annotated_frame(frame_rgb, draw_status=False)
                writer.write(annotated_frame_rgb)
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error during recording write loop: {e}")
                break
        
        writer.release()
        print(f"Video saved: {out_path}")
        # --- Analysis thread call removed ---
        
    def _get_annotated_frame(self, frame, draw_status=False):
        """Draws status info and timestamp on a copy of the frame."""
        annotated_frame = frame.copy()
        # --- YOLO annotation drawing removed ---
        
        if draw_status:
            mode_text, mode_color = ("MODE: NIGHT", (255, 255, 0)) if self.is_night_mode else ("MODE: DAY", (0, 255, 255))
            cv2.putText(annotated_frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            
            status_text = f"Monitoring: {'ON' if self.monitoring_active else 'OFF'}"
            color = (0, 255, 0) if self.monitoring_active else (100, 100, 100)
            if self.state == State.RECORDING:
                status_text, color = "Status: RECORDING", (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        with self.fps_lock:
            info_text = f"Exposure: {NIGHT_EXPOSURE_SECONDS}s" if self.is_night_mode else f"{self.fps:.1f} FPS"
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        combined_text = f"{timestamp} | {info_text}"
        cv2.putText(annotated_frame, combined_text, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    # --- get_jpeg_frame(), stop(), start_monitoring(), and stop_monitoring() are unchanged ---
    def get_jpeg_frame(self):
        """Encodes the latest streamable frame (which is now RGB) into a JPEG."""
        with self.lock:
            if self.latest_frame_for_stream is None:
                # Create a black placeholder if the camera is not ready
                frame_rgb = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame_rgb, "Initializing...", (int(FRAME_WIDTH/2)-100, int(FRAME_HEIGHT/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame_rgb = self.latest_frame_for_stream
    
        # Directly encode the RGB frame. No conversion is needed anymore.
        ret, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buffer.tobytes() if ret else None

    def stop(self):
        print("--- Shutting Down Camera System ---")
        self.running = False
        if self.state == State.RECORDING: self._stop_recording()
        if self.capture_thread.is_alive(): self.capture_thread.join(timeout=2)
        if self.processing_thread.is_alive(): self.processing_thread.join(timeout=2)
        if self.picam2: self.picam2.stop()
        print("Camera system stopped.")

    def start_monitoring(self):
        print("Monitoring has been ENABLED by user.")
        self.monitoring_active = True

    def stop_monitoring(self):
        print("Monitoring has been DISABLED by user.")
        self.monitoring_active = False
        if self.state == State.RECORDING: self._stop_recording()


#------------------------------------------
# Flask App Section
# (This section remains unchanged as its functionality is still relevant)
#------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(24)
camera = Camera()

def _list_recordings():
    items = []
    if os.path.exists(SAVE_DIR):
        for f in sorted(os.listdir(SAVE_DIR), reverse=True):
            if not f.endswith('.mp4'): continue
            fp = os.path.join(SAVE_DIR, f)
            try:
                st = os.stat(fp)
                items.append({"name": f, "mtime": st.st_mtime, "size": st.st_size})
            except FileNotFoundError: continue
    return items

def _safe_paths(filenames):
    safe = []
    base = os.path.realpath(SAVE_DIR)
    for name in filenames:
        if not name.endswith('.mp4'): continue
        ap = os.path.realpath(os.path.join(SAVE_DIR, name))
        if os.path.commonpath([ap, base]) == base and os.path.exists(ap):
            safe.append((name, ap))
    return safe

def _fmt_size(bytes_):
    for unit in ['B','KB','MB','GB','TB']:
        if bytes_ < 1024.0: return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:.1f} PB"

@app.route('/')
def index():
    recordings = _list_recordings()
    controls_html = f'''
        <form action="/stop_monitoring" method="POST">
            <button type="submit" class="control-button stop">Stop Monitoring</button>
        </form><p>Status: Monitoring for motion...</p>''' if camera.monitoring_active else '''
        <form action="/start_monitoring" method="POST">
            <button type="submit" class="control-button start">Start Monitoring</button>
        </form><p>Status: Monitoring is OFF.</p>'''

    flash_messages = ''
    messages = get_flashed_messages(with_categories=True)
    if messages:
        flash_messages = '<div class="flash-messages">'
        for category, message in messages:
            flash_messages += f'<p class="flash {category}">{message}</p>'
        flash_messages += '</div>'
    
    if not recordings:
        recordings_html = '<h2>No recordings yet.</h2>'
    else:
        rows = []
        for r in recordings:
            delete_url = url_for('delete_recording', filename=r["name"])
            rows.append(f'''
                <tr>
                    <td><input type="checkbox" name="selected" value="{r["name"]}"></td>
                    <td><a href="/recordings/{r["name"]}" target="_blank">{r["name"]}</a></td>
                    <td>{_fmt_size(r["size"])}</td>
                    <td>{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["mtime"]))}</td>
                    <td>
                        <form action="{delete_url}" method="POST" onsubmit="return confirm('Delete {r["name"]}?');">
                            <button type="submit" class="mini-btn danger">Delete</button>
                        </form>
                    </td>
                </tr>
            ''')
        rows_html = "\n".join(rows)
        recordings_html = f'''
            <h2>Recordings</h2>
            <form id="bulk-form" action="/bulk" method="POST">
                <div class="bulk-actions">
                    <label><input type="checkbox" id="select-all"> Select All</label>
                    <div class="spacer"></div>
                    <button type="button" class="control-button" onclick="submitBulk('download')">Download Selected</button>
                    <button type="button" class="control-button danger" onclick="confirmDelete()">Delete Selected</button>
                    <input type="hidden" name="action" id="bulk-action" value="">
                </div>
                <div class="table-wrap"><table class="rec-table">
                    <thead><tr><th></th><th>Filename</th><th>Size</th><th>Modified</th><th>Single</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table></div>
            </form>'''
    # The full HTML content is omitted here for brevity but is identical to your original file.
    # A placeholder is used below.
    return f'''
    <html>
        <head>
            <title>Raspberry Pi Motion Camera</title>
            <style>
                body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background-color: #1e1e1e; color: #eee; margin: 0; }}
                h1 {{ text-align: center; padding: 20px; background-color: #000; margin: 0; }}
                .stream-container {{ max-width: 90%; margin: 20px auto; border: 3px solid #444; border-radius: 8px; overflow: hidden; }}
                img {{ display: block; width: 100%; height: auto; }}
                .controls-container {{ text-align: center; margin: 20px; }}
                .controls-container p {{ color: #aaa; }}
                .control-button {{ font-size: 1em; padding: 10px 16px; border-radius: 8px; border: none; color: #fff; cursor: pointer; background:#3a84f7; }}
                .control-button.start {{ background-color: #28a745; }}
                .control-button.stop {{ background-color: #dc3545; }}
                .control-button.danger {{ background-color: #dc3545; }}
                .recordings-container {{ max-width: 90%; margin: 20px auto; padding: 10px 20px; background-color: #2a2a2a; border: 1px solid #444; border-radius: 8px; }}
                .recordings-container h2 {{ color: #eee; border-bottom: 1px solid #555; padding-bottom: 10px; }}
                .bulk-actions {{ display: flex; align-items: center; gap: 8px; margin: 12px 0; }}
                .bulk-actions .spacer {{ flex: 1; }}
                .mini-btn {{ padding: 4px 8px; border-radius: 6px; border: none; cursor: pointer; background: #555; color: #fff; }}
                .mini-btn.danger {{ background: #b72236; }}
                .table-wrap {{ overflow-x: auto; }}
                table.rec-table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
                table.rec-table th, td {{ border-bottom: 1px solid #3a3a3a; padding: 8px; text-align: left; }}
                table.rec-table th {{ color: #bbb; background: #222; position: sticky; top: 0; }}
                a {{ color: #4aa3ff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .flash-messages {{ margin: 10px 0; }}
                .flash {{ padding: 10px; border-radius: 4px; margin: 5px 0; }}
                .flash.success {{ background-color: #28a745; color: #fff; }}
                .flash.error {{ background-color: #dc3545; color: #fff; }}
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Motion Camera</h1>
            <div class="stream-container"><img src="/video_feed"></div>
            <div class="controls-container">{controls_html}</div>
            <div class="recordings-container">{flash_messages}{recordings_html}</div>
            <script>
                const selectAll = document.getElementById('select-all');
                if (selectAll) {{
                    selectAll.addEventListener('change', () => {{
                        document.querySelectorAll('input[name="selected"]').forEach(cb => cb.checked = selectAll.checked);
                    }});
                }}
                function submitBulk(action) {{
                    const form = document.getElementById('bulk-form');
                    if (![...document.querySelectorAll('input[name="selected"]')].some(cb => cb.checked)) {{
                        alert('Please select at least one video.'); return;
                    }}
                    document.getElementById('bulk-action').value = action;
                    form.submit();
                }}
                function confirmDelete() {{
                    const count = document.querySelectorAll('input[name="selected"]:checked').length;
                    if (count === 0) {{ alert('Please select at least one video.'); return; }}
                    if (confirm(`Delete ${{count}} selected video(s)? This cannot be undone.`)) submitBulk('delete');
                }}
            </script>
        </body>
    </html>
    '''

def generate_frames_for_stream(cam):
    # This function creates the placeholder image when recording is active.
    # It remains unchanged from your original file.
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    placeholder[:] = (68, 68, 68)
    text = "RECORDING IN PROGRESS"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (FRAME_WIDTH - text_size[0]) // 2
    text_y = (FRAME_HEIGHT + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, placeholder_buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    placeholder_bytes = placeholder_buffer.tobytes()

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
        safe_files = _safe_paths([filename])
        if not safe_files:
            flash(f"Invalid filename: {filename}", "error")
            return redirect(url_for('index'))
        name, file_path = safe_files[0]
        os.remove(file_path)
        flash(f"Successfully deleted {name}.", "success")
    except Exception as e:
        flash(f"Error deleting {filename}: {str(e)}", "error")
    return redirect(url_for('index'))

@app.route('/bulk', methods=['POST'])
def bulk_action():
    action = request.form.get('action', '').strip().lower()
    selected = request.form.getlist('selected')
    files = _safe_paths(selected)
    if not files: return redirect(url_for('index'))

    if action == 'delete':
        deleted_count = 0
        for name, path in files:
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {name}: {e}")
        flash(f"Deleted {deleted_count} files.", "success")
        return redirect(url_for('index'))

    elif action == 'download':
        buf = io.BytesIO()
        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_name = f"recordings_{ts}.zip"
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for name, path in files:
                zf.write(path, arcname=name)
        buf.seek(0)
        return send_file(buf, mimetype='application/zip', as_attachment=True, download_name=zip_name)
    
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
            app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down.")
    finally:
        camera.stop()
