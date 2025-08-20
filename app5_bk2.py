import cv2
import time
import os
import threading
import collections
import queue
import numpy as np
import io, zipfile
from flask import Flask, Response, send_from_directory, redirect, url_for,request, abort, send_file, flash, get_flashed_messages
from ultralytics import YOLO

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
except Exception:
    HAVE_ASTRAL = False

class State(Enum):
    IDLE = 0
    RECORDING = 1


# --- Constants ---
FRAMERATE = 30
FRAME_WIDTH = 1600
FRAME_HEIGHT = 900
JPEG_QUALITY = 95
LORES_WIDTH = 640
LORES_HEIGHT = 480
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
SUN_TRANSITION_MINUTES = 130

# Add new constants for night mode control
NIGHT_MODE_GAIN_THRESHOLD = 7.0 
DAY_MODE_GAIN_THRESHOLD = 6.0  # Lowered slightly to prevent flickering
NIGHT_EXPOSURE_SECONDS = 1
NIGHT_ANALOGUE_GAIN = 14.0

# Geo & peek scheduling
LATITUDE = 35.9211     # <-- set to your camera location
LONGITUDE = -80.5221   # <-- set to your camera location
TIMEZONE = "America/New_York"  # or your local tz

#PEEK_WINDOW_MIN = 2          # allow peeks within ± this many minutes of sunrise
#PEEK_INTERVAL_NEAR = 600.0     # seconds between peeks during the window

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
        self._last_peek_ts = 0
        self._last_mode_switch_ts = 0.0
        self.night_track_history = collections.deque(maxlen=5) 
        self.prev_lores_day_gray = None
        self.prev_lores_night_gray = None
        self.motion_history = collections.deque(maxlen=3)
        self.roi_mask = None # 

        # Stream Management
        self.stream_paused_event = threading.Event()
        self.latest_frame_for_stream = None
        self.latest_detections = []
        self.monitoring_active = False 
        self.is_night_mode = False 
        self.latest_lores_gray = None
        
        #Sunrise/Sunset Values
        self._sunrise_today = None
        self._sunset_today = None
        self._sun_refresh_after = None

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
                },
                buffer_count=2
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.max_exposure_time = self.picam2.camera_controls['ExposureTime'][1]
            print(f"Sensor max exposure time: {self.max_exposure_time} microseconds")
            time.sleep(1.0) 
            self._decide_initial_mode_by_time()
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
            # crude fallback if astral isn't installed
            now = datetime.now(tz)
            self._sunrise_today = now.replace(hour=6, minute=30, second=0, microsecond=0)
            self._sunset_today  = now.replace(hour=18, minute=30, second=0, microsecond=0)
        # next refresh: a minute after local midnight
        tomorrow = datetime.now(tz).date() + timedelta(days=1)
        self._sun_refresh_after = datetime.combine(tomorrow, datetime.min.time(), tz) + timedelta(minutes=1)

    def _decide_initial_mode_by_time(self):
        """One-shot day/night classification at startup using sunrise/sunset times."""
        try:
            # Ensure sun times are calculated first
            if not self._sunrise_today:
                self._refresh_sun_times()

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
                    if current_time != self.last_frame_time:
                        self.fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time

                main_frame = request.make_array("main")
                lores_frame = request.make_array("lores")
                lores_gray = lores_frame[0:LORES_HEIGHT, 0:LORES_WIDTH]  # Y from YUV420

                with self.lock:
                    self.pre_record_buffer.append(main_frame)
                    if self.state == State.RECORDING:
                        self.recording_queue.put(main_frame)
                    self.latest_lores_gray = lores_gray.copy()
                request.release()
                
                # The duplicated block has been removed.
                
                annotated_frame = self._get_annotated_frame(main_frame, draw_status=True)
                with self.lock:
                    self.latest_frame_for_stream = annotated_frame
            
            except Exception as e:
                print(f"Error in capture loop: {e}")
                self.running = False
        print("Capture loop has stopped.")

    def _processing_loop(self):
        """Independently captures lo-res frames to detect motion, manage state, and control camera modes."""
        frame_counter = 0

        while self.running:
            try:
                # --- Day/Night Mode Control ---
                if self.state == State.IDLE:
                    self._update_mode_by_time()

                if not self.monitoring_active:
                    if self.latest_detections:
                        self.latest_detections = []
                    time.sleep(0.1)
                    continue

                motion_detected = False
                if self.is_night_mode:
                    motion_detected = self._night_mode_airplane_detection()
                else:
                    # Use a dedicated motion detection method for day mode ---
                    motion_detected = self._day_mode_motion_detection()

                # --- YOLO processing logic (remains the same) ---
                frame_counter += 1
                if frame_counter % max(1, (FRAMERATE // YOLO_FRAME_INTERVAL)) == 0:
                    with self.lock:
                        frame_for_yolo = None if self.latest_frame_for_stream is None else self.latest_frame_for_stream.copy()
                    if frame_for_yolo is not None:
                        results = self.model.predict(frame_for_yolo, conf=YOLO_CONFIDENCE, verbose=False)
                        with self.lock:
                            self.latest_detections = results
                # --- Recording logic (remains the same) ---
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

    def _update_mode_by_time(self):
        """
        Mode maintenance based on sunrise/sunset. This consolidated version also
        handles resetting the motion detection background to prevent false triggers.
        """
        now_ts = time.time()
        if now_ts - self._last_mode_switch_ts < 60.0:
            return

        try:
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)

            if self._sun_refresh_after and now >= self._sun_refresh_after:
                print("[Sun] New day, refreshing sunrise/sunset times.")
                self._refresh_sun_times()

            day_starts = self._sunrise_today - timedelta(minutes=SUN_TRANSITION_MINUTES)
            night_starts = self._sunset_today + timedelta(minutes=SUN_TRANSITION_MINUTES)

            is_currently_daytime = day_starts <= now < night_starts

            mode_changed = False
            # Check if a switch to DAY mode is needed
            if is_currently_daytime and self.is_night_mode:
                print(f"[ModeCheck] Time ({now.strftime('%H:%M')}) is past day threshold. Switching to DAY.")
                mode_changed = True
                self._enable_day_mode()

            # Check if a switch to NIGHT mode is needed
            elif not is_currently_daytime and not self.is_night_mode:
                print(f"[ModeCheck] Time ({now.strftime('%H:%M')}) is past night threshold. Switching to NIGHT.")
                mode_changed = True
                self._enable_night_mode()

            # If a switch occurred, reset the background and pause
            if mode_changed:
                print("Mode switched. Resetting motion background and pausing for 2s to stabilize...")

                # 1. Reset background models directly
                self.prev_lores_day_gray = None
                self.prev_lores_night_gray = None
                if hasattr(self, 'motion_history'):
                    self.motion_history.clear()

                # 2. Update timestamp and pause
                self._last_mode_switch_ts = now_ts
                time.sleep(2.0) # Pause to let sensor settings stabilize

        except Exception as e:
            print(f"[ModeCheck] Error during time-based mode check: {e}")

    def _night_mode_airplane_detection(self):
        """
        A specialized detector for finding small, bright objects (like airplanes) at night.
        This version continuously tracks objects to ensure recording doesn't stop prematurely.
        """
        # --- Tuning Parameters --------------------------------------------------------------------------
        BRIGHTNESS_THRESHOLD = 38
        MAX_DISTANCE_PER_FRAME = 90
        MIN_CONSECUTIVE_FRAMES = 3
        MIN_SPOT_AREA = 2
        MAX_SPOT_AREA = 500

        with self.lock:
            lores_gray = None if self.latest_lores_gray is None else self.latest_lores_gray.copy()
        if lores_gray is None:
            return False

        # 1. Find all potential bright spots in the current frame
        _, thresh = cv2.threshold(lores_gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_spots = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_SPOT_AREA <= area <= MAX_SPOT_AREA:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    current_spots.append({'pos': (cx, cy), 'track_len': 1})
        
        # 2. Compare with history to build tracks
        if len(self.night_track_history) > 0:
            prev_spots = self.night_track_history[-1]
            for i in range(len(current_spots)):
                for prev_spot in prev_spots:
                    dist = np.linalg.norm(np.array(current_spots[i]['pos']) - np.array(prev_spot['pos']))
                    if dist < MAX_DISTANCE_PER_FRAME:
                        current_spots[i]['track_len'] = prev_spot['track_len'] + 1
                        break
        
        self.night_track_history.append(current_spots)

        # 3. Check if any track is long enough to be considered motion.
        motion_found = False
        for spot in current_spots:
            # Only print the confirmation message ONCE when a track is first established.
            if spot['track_len'] == MIN_CONSECUTIVE_FRAMES:
                print(f"✔️ Confirmed new track of length {spot['track_len']} at {spot['pos']}.")
            
            # If any track is long enough, we consider it ongoing motion.
            if spot['track_len'] >= MIN_CONSECUTIVE_FRAMES:
                motion_found = True

        # By NOT clearing the history, we allow tracks to persist frame after frame.
        # This will continuously return True as long as the object is tracked.
        return motion_found

    def _enable_night_mode(self, force_revert=False):
        """Lock long exposure / high gain and low framerate for night viewing."""
        # ensure we have the clamp
        if not hasattr(self, 'max_exposure_time'):
            self.max_exposure_time = self.picam2.camera_controls['ExposureTime'][1]

        if not force_revert:
            try:
                g = self.picam2.capture_metadata().get("AnalogueGain", 0.0)
                print(f"🌙 Switching to NIGHT MODE. Analog gain: {float(g):.2f}")
            except Exception:
                print("🌙 Switching to NIGHT MODE.")

        exposure_time_us = int(NIGHT_EXPOSURE_SECONDS * 1_000_000)
        exposure_time_us = min(exposure_time_us, int(self.max_exposure_time))

        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exposure_time_us,
            "AnalogueGain": float(NIGHT_ANALOGUE_GAIN),
            "FrameRate": 1/NIGHT_EXPOSURE_SECONDS,  # allow long exposure
        })
        self.is_night_mode = True
        if not force_revert:
            print(f"    -> Set Exposure: {exposure_time_us/1_000_000:.2f}s, Gain: {NIGHT_ANALOGUE_GAIN}")


    def _enable_day_mode(self):
        """Return to normal auto-exposure/framerate."""
        print("☀️ Switching to DAY MODE.")
        # smooth transition
        self.picam2.set_controls({"AeEnable": False, "AwbEnable": False, "FrameRate": FRAMERATE})
        time.sleep(0.3)
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True, "FrameRate": FRAMERATE})
        self.is_night_mode = False
        time.sleep(0.7)  # let AE converge

    def _day_mode_motion_detection(self):
        """
        Detects motion in day mode, now with more robust handling of the background state.
        """
        with self.lock:
            lores_gray = None if self.latest_lores_gray is None else self.latest_lores_gray.copy()
        if lores_gray is None:
            return False

        lores_gray_blurred = cv2.GaussianBlur(lores_gray, (9, 9), 0)

        # This check is now the key to handling the mode-switch reset.
        # If the background is None (because it was just reset), we initialize it and exit for this frame.
        if self.prev_lores_day_gray is None:
            self.prev_lores_day_gray = lores_gray_blurred
            self.motion_history.clear()
            return False

        # If the background is valid, we proceed with motion detection.
        frame_delta = cv2.absdiff(self.prev_lores_day_gray, lores_gray_blurred)
        self.prev_lores_day_gray = lores_gray_blurred

        if self.roi_mask is not None:
            frame_delta = cv2.bitwise_and(frame_delta, frame_delta, mask=self.roi_mask)

        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 2
        max_area = 3000
        motion_detected = False
        for c in contours:
            if min_area < cv2.contourArea(c) < max_area:
                motion_detected = True
                break

        self.motion_history.append(motion_detected)
        
        if len(self.motion_history) == self.motion_history.maxlen:
            if sum(self.motion_history) >= 2:
                return True

        return False

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

    def _stop_recording(self):
        """Signals the recording thread to stop and waits for it to finish."""
        with self.lock:
            if self.state == State.IDLE:
                return
            print("Stopping recording and finalizing video file...")
            self.state = State.IDLE  # Signal the thread to stop its loop
            self.current_recording_path = None

        if self.recording_thread and self.recording_thread.is_alive():
            print("Waiting for recording thread to finish...")
            # We wait longer here to allow the queue to drain properly.
            self.recording_thread.join(timeout=30.0) 
            if self.recording_thread.is_alive():
                print("Error: Recording thread is unresponsive.")
            else:
                print("Recording thread finished.")
        
        self.stream_paused_event.clear()

    def _night_mode_motion_detection(self):
        """
        Detects motion in night mode by looking for changes in a specific frame of the lo-res stream.
        This is a different approach from the day mode's pixel count to accommodate the long exposures.
        """
        with self.lock:
            lores_gray = None if self.latest_lores_gray is None else self.latest_lores_gray.copy()
        if lores_gray is None:
            return False

        lores_gray = cv2.GaussianBlur(lores_gray, (15, 15), 0)
        if not hasattr(self, 'prev_lores_night_gray'):
            self.prev_lores_night_gray = lores_gray
            return False
        frame_delta = cv2.absdiff(self.prev_lores_night_gray, lores_gray)
        self.prev_lores_night_gray = lores_gray

        thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 5  # Adjust this value based on testing
        for c in contours:
            if cv2.contourArea(c) > min_area:
                print("Night motion detected!")
                return True
            
            return False

    def _recording_loop(self):
        """Writes frames from the queue to the video file and then triggers analysis."""
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
            except Exception as e:
                print(f"Error during recording write loop: {e}")
                break
        
        # --- This is the crucial part of the new logic ---
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved: {out_path}")
            
            # Now that the file is closed and valid, start the analysis.
            print(f"Queueing video for analysis: {os.path.basename(out_path)}")
            analysis_thread = threading.Thread(
                target=self._analyze_and_rename_video,
                args=(out_path,),
                daemon=True
            )
            analysis_thread.start()
        
        self.video_writer = None

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

from flask import Flask, Response, send_from_directory, redirect, url_for, request, abort, send_file
import io, zipfile

app = Flask(__name__)
app.secret_key = os.urandom(24)
camera = Camera()

def _list_recordings():
    items = []
    if os.path.exists(SAVE_DIR):
        for f in os.listdir(SAVE_DIR):
            if not f.endswith('.mp4'):
                continue
            fp = os.path.join(SAVE_DIR, f)
            try:
                st = os.stat(fp)
            except FileNotFoundError:
                continue
            items.append({
                "name": f,
                "mtime": st.st_mtime,
                "size": st.st_size
            })
    # newest first
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

def _safe_paths(filenames):
    """Return list of (name, absolute_path) for valid files under SAVE_DIR."""
    safe = []
    base = os.path.realpath(SAVE_DIR)
    for name in filenames:
        if not name.endswith('.mp4'):
            continue
        ap = os.path.realpath(os.path.join(SAVE_DIR, name))
        # ensure inside SAVE_DIR and exists
        try:
            if os.path.commonpath([ap, base]) == base and os.path.exists(ap):
                safe.append((name, ap))
        except Exception:
            # commonpath can raise if mix of drives, ignore bad entries
            continue
    return safe

def _fmt_size(bytes_):
    for unit in ['B','KB','MB','GB','TB']:
        if bytes_ < 1024.0:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:.1f} PB"

@app.route('/')
def index():
    """Serves the main HTML page and lists recordings with bulk actions."""
    recordings = _list_recordings()
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
    # Build flash messages HTML
    flash_messages = ''
    messages = get_flashed_messages(with_categories=True)
    if messages:
        flash_messages = '<div class="flash-messages">'
        for category, message in messages:
            flash_messages += f'<p class="flash {category}">{message}</p>'
        flash_messages += '</div>'
    
    # Build recordings list with checkboxes
    if not recordings:
        recordings_html = '<h2>No recordings yet.</h2>'
    else:
        rows = []
        for r in recordings:
            # Use url_for to ensure proper URL encoding
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
                <div class="table-wrap">
                    <table class="rec-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>Filename</th>
                                <th>Size</th>
                                <th>Modified</th>
                                <th>Single</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </form>
        '''
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
                .control-button:disabled {{ opacity: .6; cursor: not-allowed; }}
                .recordings-container {{ max-width: 90%; margin: 20px auto; padding: 10px 20px; background-color: #2a2a2a; border: 1px solid #444; border-radius: 8px; }}
                .recordings-container h2 {{ color: #eee; border-bottom: 1px solid #555; padding-bottom: 10px; }}
                .bulk-actions {{ display: flex; align-items: center; gap: 8px; margin: 12px 0; }}
                .bulk-actions .spacer {{ flex: 1; }}
                .mini-btn {{ padding: 4px 8px; border-radius: 6px; border: none; cursor: pointer; background: #555; color: #fff; }}
                .mini-btn.danger {{ background: #b72236; }}
                .table-wrap {{ overflow-x: auto; }}
                table.rec-table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
                table.rec-table th, table.rec-table td {{ border-bottom: 1px solid #3a3a3a; padding: 8px; text-align: left; }}
                table.rec-table th {{ color: #bbb; background: #222; position: sticky; top: 0; z-index: 1; }}
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
            <div class="stream-container">
                <img src="/video_feed">
            </div>
            <div class="controls-container">
                {controls_html}
            </div>
            <div class="recordings-container">
                {flash_messages}
                {recordings_html}
            </div>
            <script>
                // Select All toggle
                const selectAll = document.getElementById('select-all');
                if (selectAll) {{
                    selectAll.addEventListener('change', () => {{
                        document.querySelectorAll('input[name="selected"]').forEach(cb => cb.checked = selectAll.checked);
                    }});
                }}
                function submitBulk(action) {{
                    const form = document.getElementById('bulk-form');
                    const anyChecked = [...document.querySelectorAll('input[name="selected"]')].some(cb => cb.checked);
                    if (!anyChecked) {{
                        alert('Please select at least one video.');
                        return;
                    }}
                    document.getElementById('bulk-action').value = action;
                    form.submit();
                }}
                function confirmDelete() {{
                    const count = [...document.querySelectorAll('input[name="selected"]:checked')].length;
                    if (count === 0) {{
                        alert('Please select at least one video.');
                        return;
                    }}
                    if (confirm(`Delete ${{count}} selected video(s)? This cannot be undone.`)) {{
                        submitBulk('delete');
                    }}
                }}
            </script>
        </body>
    </html>
    '''

def generate_frames_for_stream(cam):
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    placeholder[:] = (68, 68, 68)  # BGR
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
    print(f"Attempting to delete single file: {filename}")
    try:
        # Use _safe_paths for consistent path validation
        safe_files = _safe_paths([filename])
        if not safe_files:
            print(f"Invalid or unsafe filename: {filename}")
            flash(f"Invalid filename: {filename}", "error")
            return redirect(url_for('index'))
        name, file_path = safe_files[0]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            flash(f"File {name} not found.", "error")
            return redirect(url_for('index'))
        if not os.access(file_path, os.W_OK):
            print(f"No write permission for file: {file_path}")
            flash(f"Cannot delete {name}: Permission denied.", "error")
            return redirect(url_for('index'))
        for attempt in range(3):  # Retry up to 3 times
            try:
                os.remove(file_path)
                print(f"Deleted recording: {name} on attempt {attempt + 1}")
                flash(f"Successfully deleted {name}.", "success")
                break
            except OSError as e:
                if e.errno != 13:  # Not a permission error
                    raise
                print(f"Retry {attempt + 1}/3: Cannot delete {name}: {e}")
                time.sleep(1.0)  # Wait 1s before retry
        else:
            print(f"Failed to delete {name} after 3 retries")
            flash(f"Cannot delete {name}: File may be in use.", "error")
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")
        flash(f"Error deleting {filename}: {str(e)}", "error")
    return redirect(url_for('index'))

@app.route('/bulk', methods=['POST'])
def bulk_action():
    action = request.form.get('action', '').strip().lower()
    selected = request.form.getlist('selected')
    files = _safe_paths(selected)

    if not files:
        return redirect(url_for('index'))

    if action == 'delete':
        deleted = 0
        for name, path in files:
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"Error deleting {name}: {e}")
        print(f"Bulk delete: removed {deleted} file(s).")
        return redirect(url_for('index'))

    elif action == 'download':
        # Create a zip in memory (good for modest selections)
        buf = io.BytesIO()
        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_name = f"recordings_{ts}.zip"
        try:
            with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for name, path in files:
                    # store with plain filename
                    zf.write(path, arcname=name)
            buf.seek(0)
            return send_file(
                buf,
                mimetype='application/zip',
                as_attachment=True,
                download_name=zip_name
            )
        except Exception as e:
            print(f"Error creating zip: {e}")
            abort(500)
    else:
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
            app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
        else:
            print("\n--- Application startup failed: Could not initialize camera. ---")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down.")
    finally:
        camera.stop()
