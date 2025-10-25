import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import sounddevice as sd
from scipy.io import wavfile
import tempfile
import os

# Page config
st.set_page_config(
    page_title="CycleSafe AI",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
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
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .camera-feed {
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        border: 3px solid #667eea;
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

class CameraDangerDetector:
    """Detects vehicles and obstacles using OpenCV cascade classifiers"""
    
    def __init__(self):
        # Load pre-trained cascades
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        self.danger_threshold = 0.3  # Distance threshold for danger
        
    def detect_dangers(self, frame):
        """Detect vehicles and potential dangers"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 3)
        
        dangers = []
        for (x, y, w, h) in cars:
            # Calculate relative size (closer objects are larger)
            frame_area = frame.shape[0] * frame.shape[1]
            car_area = w * h
            relative_size = car_area / frame_area
            
            danger_level = 'low'
            if relative_size > 0.3:
                danger_level = 'high'
            elif relative_size > 0.15:
                danger_level = 'medium'
            
            dangers.append({
                'type': 'vehicle',
                'bbox': (x, y, w, h),
                'level': danger_level,
                'size': relative_size
            })
            
            # Draw bounding box
            color = (0, 0, 255) if danger_level == 'high' else (0, 165, 255) if danger_level == 'medium' else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{danger_level.upper()}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, dangers

class AudioDangerDetector:
    """Detects dangerous sounds using volume and frequency analysis"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.danger_keywords = ['horn', 'siren', 'loud']
        
    def analyze_audio(self, audio_data):
        """Analyze audio for dangerous sounds"""
        # Calculate volume (RMS)
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Simple threshold-based detection
        danger_detected = False
        danger_type = None
        
        if rms > 0.3:  # High volume threshold
            danger_detected = True
            danger_type = 'loud_noise'
        
        # Frequency analysis for horn detection
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        
        # Horn frequencies typically 400-600 Hz
        horn_range = np.where((freqs > 400) & (freqs < 600))
        if len(horn_range[0]) > 0:
            horn_energy = np.sum(np.abs(fft[horn_range]))
            if horn_energy > 1000:
                danger_detected = True
                danger_type = 'horn_detected'
        
        return danger_detected, danger_type, rms

def process_camera(camera_id, position, detector, frame_queue, alert_queue):
    """Process camera feed in separate thread"""
    cap = cv2.VideoCapture(camera_id)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect dangers
        processed_frame, dangers = detector.detect_dangers(frame)
        
        # Add position label
        cv2.putText(processed_frame, f'{position.upper()} CAMERA', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Send to display queue
        frame_queue.put((position, processed_frame))
        
        # Check for high-level dangers
        high_dangers = [d for d in dangers if d['level'] == 'high']
        if high_dangers:
            alert_queue.put({
                'time': datetime.now().strftime('%H:%M:%S'),
                'position': position,
                'type': 'vehicle_close',
                'level': 'high',
                'message': f'⚠️ DANGER: Vehicle approaching from {position}!'
            })
            st.session_state.detection_count[position] += 1
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()

def record_and_analyze_audio(detector, alert_queue):
    """Record and analyze audio continuously"""
    duration = 2  # seconds per chunk
    
    while st.session_state.running:
        # Record audio
        audio_data = sd.rec(int(duration * detector.sample_rate), 
                           samplerate=detector.sample_rate, channels=1)
        sd.wait()
        audio_data = audio_data.flatten()
        
        # Analyze
        danger_detected, danger_type, rms = detector.analyze_audio(audio_data)
        
        if danger_detected:
            alert_queue.put({
                'time': datetime.now().strftime('%H:%M:%S'),
                'position': 'audio',
                'type': danger_type,
                'level': 'medium',
                'message': f'🔊 Audio Alert: {danger_type.replace("_", " ").title()}'
            })
            st.session_state.detection_count['audio'] += 1

# Main UI
st.markdown('<div class="main-header"><h1>🚴 CycleSafe AI</h1><p>Advanced Cycling Safety Monitoring System</p></div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ System Controls")
    
    # Camera selection
    st.subheader("📹 Camera Setup")
    front_cam = st.number_input("Front Camera ID", 0, 10, 0)
    rear_cam = st.number_input("Rear Camera ID", 0, 10, 1)
    
    st.subheader("🎤 Audio Setup")
    enable_audio = st.checkbox("Enable Audio Monitoring", value=True)
    
    st.divider()
    
    # Start/Stop button
    if not st.session_state.running:
        if st.button("🚀 Start Monitoring", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Monitoring", type="secondary", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Detection Stats")
    st.metric("Front Camera", st.session_state.detection_count['front'])
    st.metric("Rear Camera", st.session_state.detection_count['rear'])
    st.metric("Audio Alerts", st.session_state.detection_count['audio'])

# Main content area
if st.session_state.running:
    # Create queues
    frame_queue = queue.Queue(maxsize=10)
    alert_queue = queue.Queue()
    
    # Initialize detectors
    camera_detector = CameraDangerDetector()
    audio_detector = AudioDangerDetector()
    
    # Start threads
    front_thread = threading.Thread(target=process_camera, 
                                    args=(front_cam, 'front', camera_detector, frame_queue, alert_queue))
    rear_thread = threading.Thread(target=process_camera, 
                                   args=(rear_cam, 'rear', camera_detector, frame_queue, alert_queue))
    
    front_thread.daemon = True
    rear_thread.daemon = True
    front_thread.start()
    rear_thread.start()
    
    if enable_audio:
        audio_thread = threading.Thread(target=record_and_analyze_audio, 
                                       args=(audio_detector, alert_queue))
        audio_thread.daemon = True
        audio_thread.start()
    
    # Alert section
    alert_col = st.container()
    with alert_col:
        alert_placeholder = st.empty()
    
    # Camera feeds
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 Front Camera")
        front_placeholder = st.empty()
    with col2:
        st.subheader("📹 Rear Camera")
        rear_placeholder = st.empty()
    
    # Processing loop
    while st.session_state.running:
        # Check for new alerts
        while not alert_queue.empty():
            alert = alert_queue.get()
            st.session_state.alerts.insert(0, alert)
            st.session_state.alerts = st.session_state.alerts[:5]  # Keep last 5
        
        # Display alerts
        if st.session_state.alerts:
            latest_alert = st.session_state.alerts[0]
            alert_class = 'danger' if latest_alert['level'] == 'high' else 'warning'
            alert_placeholder.markdown(
                f'<div class="alert-box {alert_class}"><h3>{latest_alert["message"]}</h3><p>Time: {latest_alert["time"]}</p></div>',
                unsafe_allow_html=True
            )
        else:
            alert_placeholder.markdown(
                '<div class="alert-box safe"><h3>✅ All Clear</h3><p>No immediate dangers detected</p></div>',
                unsafe_allow_html=True
            )
        
        # Update camera feeds
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
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>🎯 Ready to Start</h2>
        <p style='font-size: 1.2rem; color: #666;'>
            Configure your cameras and audio settings in the sidebar, then click <strong>Start Monitoring</strong> to begin.
        </p>
        <div style='margin-top: 2rem;'>
            <h3>Features:</h3>
            <ul style='text-align: left; display: inline-block;'>
                <li>🎥 Dual camera monitoring (front & rear)</li>
                <li>🤖 AI-powered vehicle detection</li>
                <li>🔊 Audio danger detection</li>
                <li>⚡ Real-time alerts</li>
                <li>📊 Detection statistics</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
