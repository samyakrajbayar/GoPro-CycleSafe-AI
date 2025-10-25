import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import sounddevice as sd
import requests
import json
from urllib.parse import urljoin

# Page config
st.set_page_config(
    page_title="GoPro CycleSafe AI",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #00A8E8 0%, #007EA7 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .gopro-badge {
        display: inline-block;
        background: #fff;
        color: #00A8E8;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    .danger {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-left: 5px solid #c0392b;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
    }
    .warning {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: #2d3436;
        border-left: 5px solid #f39c12;
    }
    .safe {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
        border-left: 5px solid #27ae60;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #00A8E8;
    }
    .camera-feed {
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        border: 3px solid #00A8E8;
    }
    .connection-status {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        display: inline-block;
        font-weight: bold;
    }
    .connected {
        background: #2ecc71;
        color: white;
    }
    .disconnected {
        background: #e74c3c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = {'front': 0, 'rear': 0, 'audio': 0}
if 'gopro_status' not in st.session_state:
    st.session_state.gopro_status = {'front': False, 'rear': False}

class GoProCamera:
    """Interface for GoPro cameras via WiFi/USB"""
    
    def __init__(self, ip_address, name="GoPro"):
        self.ip = ip_address
        self.name = name
        self.base_url = f"http://{ip_address}:8080"
        self.stream_url = None
        self.connected = False
        
    def connect(self):
        """Connect to GoPro and start preview"""
        try:
            # Wake up GoPro
            response = requests.get(f"{self.base_url}/gp/gpControl/command/system/sleep?p=0", timeout=5)
            
            # Start preview (webcam mode for GoPro 8+)
            requests.get(f"{self.base_url}/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart", timeout=5)
            
            # Set stream URL (GoPro streams via UDP or HTTP depending on model)
            # For GoPro Hero 8+, use webcam mode
            self.stream_url = f"udp://@:{self.ip}:8554"
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to {self.name}: {e}")
            self.connected = False
            return False
    
    def get_stream(self):
        """Get video stream from GoPro"""
        if not self.connected:
            return None
        
        # For GoPro webcam mode, use direct connection
        # Most GoPros stream at port 8554 or can be accessed via USB
        cap = cv2.VideoCapture(self.stream_url)
        
        if not cap.isOpened():
            # Fallback: Try direct USB connection
            # GoPros appear as video devices when connected via USB
            cap = cv2.VideoCapture(f"/dev/video{self.ip.split('.')[-1]}")  
        
        return cap
    
    def set_settings(self, resolution="1080p", fps=30, fov="wide"):
        """Configure GoPro settings for optimal detection"""
        try:
            # Set resolution
            res_map = {"1080p": 9, "720p": 7, "480p": 5}
            requests.get(f"{self.base_url}/gp/gpControl/setting/2/{res_map.get(resolution, 9)}", timeout=3)
            
            # Set FPS
            fps_map = {30: 5, 60: 8, 120: 11}
            requests.get(f"{self.base_url}/gp/gpControl/setting/3/{fps_map.get(fps, 5)}", timeout=3)
            
            # Set FOV
            fov_map = {"wide": 0, "medium": 1, "narrow": 2, "linear": 4}
            requests.get(f"{self.base_url}/gp/gpControl/setting/121/{fov_map.get(fov, 4)}", timeout=3)
            
            return True
        except:
            return False
    
    def disconnect(self):
        """Disconnect from GoPro"""
        try:
            requests.get(f"{self.base_url}/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=stop", timeout=3)
            self.connected = False
        except:
            pass

class EnhancedDangerDetector:
    """Enhanced detection for cycling with GoPro footage"""
    
    def __init__(self):
        # Load multiple cascades for better detection
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # YOLOv3-tiny for better object detection (if available)
        try:
            self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
            self.classes = open("coco.names").read().strip().split("\n")
            self.yolo_available = True
        except:
            self.yolo_available = False
            print("YOLO not available, using cascade classifiers")
    
    def detect_motion(self, frame):
        """Detect significant motion in frame"""
        fg_mask = self.bg_subtractor.apply(frame)
        motion_pixels = np.sum(fg_mask == 255)
        frame_pixels = frame.shape[0] * frame.shape[1]
        motion_ratio = motion_pixels / frame_pixels
        
        return motion_ratio > 0.05, motion_ratio
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO"""
        if not self.yolo_available:
            return []
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.classes[class_id] in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detections.append({
                        'type': self.classes[class_id],
                        'bbox': (x, y, w, h),
                        'confidence': float(confidence)
                    })
        
        return detections
    
    def detect_dangers(self, frame):
        """Comprehensive danger detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Motion detection
        has_motion, motion_ratio = self.detect_motion(frame)
        
        # Try YOLO first
        if self.yolo_available:
            detections = self.detect_with_yolo(frame)
        else:
            # Fallback to cascade
            cars = self.car_cascade.detectMultiScale(gray, 1.1, 3)
            detections = []
            for (x, y, w, h) in cars:
                detections.append({
                    'type': 'vehicle',
                    'bbox': (x, y, w, h),
                    'confidence': 0.7
                })
        
        # Analyze detections
        dangers = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in detections:
            x, y, w, h = det['bbox']
            obj_area = w * h
            relative_size = obj_area / frame_area
            
            # Determine danger level
            if relative_size > 0.35:
                danger_level = 'critical'
                color = (0, 0, 255)  # Red
            elif relative_size > 0.2:
                danger_level = 'high'
                color = (0, 100, 255)  # Orange
            elif relative_size > 0.1:
                danger_level = 'medium'
                color = (0, 165, 255)  # Light orange
            else:
                danger_level = 'low'
                color = (0, 255, 0)  # Green
            
            dangers.append({
                'type': det['type'],
                'bbox': (x, y, w, h),
                'level': danger_level,
                'size': relative_size,
                'confidence': det.get('confidence', 0.7)
            })
            
            # Draw detection
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            label = f"{det['type'].upper()}: {danger_level.upper()}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add motion indicator
        if has_motion:
            cv2.putText(frame, f"MOTION: {motion_ratio:.2%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame, dangers

class AudioDangerDetector:
    """Enhanced audio detection for cycling environment"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.horn_freq_range = (300, 700)  # Hz
        self.siren_freq_range = (800, 1500)  # Hz
        
    def analyze_audio(self, audio_data):
        """Comprehensive audio analysis"""
        # Volume analysis
        rms = np.sqrt(np.mean(audio_data**2))
        
        # FFT for frequency analysis
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        magnitudes = np.abs(fft)
        
        dangers = []
        
        # Loud noise detection
        if rms > 0.25:
            dangers.append(('loud_noise', 'high', rms))
        
        # Horn detection
        horn_indices = np.where((freqs > self.horn_freq_range[0]) & 
                                (freqs < self.horn_freq_range[1]))[0]
        if len(horn_indices) > 0:
            horn_energy = np.sum(magnitudes[horn_indices])
            if horn_energy > 800:
                dangers.append(('horn', 'critical', horn_energy))
        
        # Siren detection
        siren_indices = np.where((freqs > self.siren_freq_range[0]) & 
                                 (freqs < self.siren_freq_range[1]))[0]
        if len(siren_indices) > 0:
            siren_energy = np.sum(magnitudes[siren_indices])
            if siren_energy > 1000:
                dangers.append(('siren', 'critical', siren_energy))
        
        return dangers, rms

def process_gopro_camera(gopro, position, detector, frame_queue, alert_queue):
    """Process GoPro camera feed"""
    cap = gopro.get_stream()
    
    if cap is None or not cap.isOpened():
        st.session_state.gopro_status[position] = False
        return
    
    st.session_state.gopro_status[position] = True
    frame_count = 0
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Process every frame for cycling safety
        processed_frame, dangers = detector.detect_dangers(frame)
        
        # Add GoPro branding and info
        cv2.putText(processed_frame, f'{position.upper()} - {gopro.name}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed_frame, f'Frame: {frame_count}', (10, processed_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Send to display
        frame_queue.put((position, processed_frame))
        
        # Check for critical/high dangers
        critical_dangers = [d for d in dangers if d['level'] in ['critical', 'high']]
        if critical_dangers:
            for danger in critical_dangers:
                alert_queue.put({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'position': position,
                    'type': danger['type'],
                    'level': danger['level'],
                    'message': f'{"🚨" if danger["level"]=="critical" else "⚠️"} {danger["level"].upper()}: {danger["type"]} detected from {position}!'
                })
            st.session_state.detection_count[position] += 1
        
        time.sleep(0.02)  # ~50 FPS processing
    
    cap.release()
    gopro.disconnect()

def record_and_analyze_audio(detector, alert_queue):
    """Enhanced audio monitoring"""
    duration = 1.5
    
    while st.session_state.running:
        audio_data = sd.rec(int(duration * detector.sample_rate), 
                           samplerate=detector.sample_rate, channels=1)
        sd.wait()
        audio_data = audio_data.flatten()
        
        dangers, rms = detector.analyze_audio(audio_data)
        
        for danger_type, level, intensity in dangers:
            alert_queue.put({
                'time': datetime.now().strftime('%H:%M:%S'),
                'position': 'audio',
                'type': danger_type,
                'level': level,
                'message': f'🔊 {level.upper()}: {danger_type.replace("_", " ").title()} detected!'
            })
            st.session_state.detection_count['audio'] += 1

# Main UI
st.markdown('''
<div class="main-header">
    <h1>🚴 GoPro CycleSafe AI</h1>
    <p>Professional Cycling Safety with Dual GoPro Cameras</p>
    <span class="gopro-badge">POWERED BY GOPRO</span>
</div>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ GoPro Configuration")
    
    st.subheader("📹 Front GoPro")
    front_ip = st.text_input("Front GoPro IP", "10.5.5.9")
    front_name = st.text_input("Front Camera Name", "GoPro Hero 11")
    
    st.subheader("📹 Rear GoPro")
    rear_ip = st.text_input("Rear GoPro IP", "10.5.5.10")
    rear_name = st.text_input("Rear Camera Name", "GoPro Hero 10")
    
    st.divider()
    
    st.subheader("🎥 Camera Settings")
    resolution = st.selectbox("Resolution", ["1080p", "720p", "480p"], index=0)
    fps = st.selectbox("FPS", [30, 60, 120], index=0)
    fov = st.selectbox("FOV", ["linear", "wide", "medium", "narrow"], index=0)
    
    st.subheader("🎤 Audio Settings")
    enable_audio = st.checkbox("Enable Audio Monitoring", value=True)
    
    st.divider()
    
    # Connection status
    st.subheader("📡 Connection Status")
    front_status = "🟢 Connected" if st.session_state.gopro_status.get('front', False) else "🔴 Disconnected"
    rear_status = "🟢 Connected" if st.session_state.gopro_status.get('rear', False) else "🔴 Disconnected"
    st.markdown(f"**Front:** {front_status}")
    st.markdown(f"**Rear:** {rear_status}")
    
    st.divider()
    
    # Start/Stop
    if not st.session_state.running:
        if st.button("🚀 Start Monitoring", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Monitoring", type="secondary", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    
    st.divider()
    
    st.subheader("📊 Detection Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Front", st.session_state.detection_count['front'], delta_color="inverse")
        st.metric("Audio", st.session_state.detection_count['audio'], delta_color="inverse")
    with col2:
        st.metric("Rear", st.session_state.detection_count['rear'], delta_color="inverse")
        total = sum(st.session_state.detection_count.values())
        st.metric("Total", total, delta_color="inverse")

# Main content
if st.session_state.running:
    # Initialize
    frame_queue = queue.Queue(maxsize=10)
    alert_queue = queue.Queue()
    
    # Create GoPro instances
    front_gopro = GoProCamera(front_ip, front_name)
    rear_gopro = GoProCamera(rear_ip, rear_name)
    
    # Connect and configure
    with st.spinner("Connecting to GoPro cameras..."):
        front_gopro.connect()
        front_gopro.set_settings(resolution, fps, fov)
        
        rear_gopro.connect()
        rear_gopro.set_settings(resolution, fps, fov)
    
    # Initialize detectors
    danger_detector = EnhancedDangerDetector()
    audio_detector = AudioDangerDetector()
    
    # Start threads
    threads = []
    
    front_thread = threading.Thread(target=process_gopro_camera,
                                    args=(front_gopro, 'front', danger_detector, frame_queue, alert_queue))
    rear_thread = threading.Thread(target=process_gopro_camera,
                                   args=(rear_gopro, 'rear', danger_detector, frame_queue, alert_queue))
    
    front_thread.daemon = True
    rear_thread.daemon = True
    threads.extend([front_thread, rear_thread])
    
    if enable_audio:
        audio_thread = threading.Thread(target=record_and_analyze_audio,
                                       args=(audio_detector, alert_queue))
        audio_thread.daemon = True
        threads.append(audio_thread)
    
    for t in threads:
        t.start()
    
    # UI Layout
    alert_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📹 Front - {front_name}")
        front_placeholder = st.empty()
    with col2:
        st.subheader(f"📹 Rear - {rear_name}")
        rear_placeholder = st.empty()
    
    # Processing loop
    while st.session_state.running:
        # Handle alerts
        while not alert_queue.empty():
            alert = alert_queue.get()
            st.session_state.alerts.insert(0, alert)
            st.session_state.alerts = st.session_state.alerts[:3]
        
        # Display latest alert
        if st.session_state.alerts:
            latest = st.session_state.alerts[0]
            alert_class = 'danger' if latest['level'] in ['critical', 'high'] else 'warning'
            alert_placeholder.markdown(
                f'''<div class="alert-box {alert_class}">
                <h2>{latest["message"]}</h2>
                <p><strong>Time:</strong> {latest["time"]} | <strong>Source:</strong> {latest["position"].upper()}</p>
                </div>''',
                unsafe_allow_html=True
            )
        else:
            alert_placeholder.markdown(
                '<div class="alert-box safe"><h2>✅ All Clear</h2><p>No immediate dangers detected</p></div>',
                unsafe_allow_html=True
            )
        
        # Update feeds
        try:
            position, frame = frame_queue.get(timeout=1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if position == 'front':
                front_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)
            else:
                rear_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)
        except queue.Empty:
            pass
        
        time.sleep(0.01)

else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>🎯 GoPro Setup Instructions</h2>
        <div style='text-align: left; max-width: 800px; margin: 2rem auto;'>
            <h3>📱 Step 1: Prepare Your GoPros</h3>
            <ol>
                <li>Turn on both GoPro cameras</li>
                <li>Enable WiFi on each camera (Settings → Connections → WiFi)</li>
                <li>Note the IP address (usually 10.5.5.9 for first camera)</li>
                <li>For USB connection: Connect via USB-C and enable GoPro Webcam mode</li>
            </ol>
            
            <h3>🔧 Step 2: Configure Settings</h3>
            <ol>
                <li>Enter IP addresses in the sidebar</li>
                <li>Choose resolution (1080p recommended for balance)</li>
                <li>Set FPS (30fps for detection, 60fps for smoother video)</li>
                <li>Select FOV (Linear recommended for minimal distortion)</li>
            </ol>
            
            <h3>🚀 Step 3: Start Monitoring</h3>
            <ol>
                <li>Click <strong>Start Monitoring</strong></li>
                <li>System will connect to both cameras</li>
                <li>Real-time detection begins automatically</li>
            </ol>
            
            <h3>✨ Features:</h3>
            <ul>
                <li>🎥 Dual GoPro camera support (WiFi or USB)</li>
                <li>🤖 Advanced AI vehicle detection (YOLO + Cascades)</li>
                <li>🎯 Motion detection and tracking</li>
                <li>🔊 Audio danger detection (horns, sirens)</li>
                <li>⚡ Real-time critical alerts</li>
                <li>📊 Comprehensive statistics</li>
                <li>🎬 Professional GoPro integration</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>GoPro CycleSafe AI</strong> - Professional cycling safety monitoring system</p>
    <p>Supports GoPro Hero 8, 9, 10, 11, 12 and newer models</p>
</div>
""", unsafe_allow_html=True)
