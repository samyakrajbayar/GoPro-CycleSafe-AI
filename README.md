# 🚴 GoPro CycleSafe AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Professional AI-Powered Cycling Safety System with Dual GoPro Camera Support**

Real-time vehicle detection • Audio danger alerts • Beautiful web interface

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [Hardware Requirements](#-hardware-requirements)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**GoPro CycleSafe AI** is an advanced cycling safety monitoring system that uses dual GoPro cameras (front and rear) combined with AI-powered object detection and audio analysis to keep cyclists safe on the road. The system provides real-time alerts for approaching vehicles, dangerous sounds, and potential hazards.

### Why CycleSafe AI?

- 🚗 **Real-time Vehicle Detection**: Instantly detects cars, trucks, buses, and motorcycles
- 🎯 **Distance Estimation**: Calculates danger levels based on object proximity
- 🔊 **Audio Monitoring**: Detects horns, sirens, and loud noises
- 📹 **Dual Camera Support**: Monitors both front and rear simultaneously
- ⚡ **Instant Alerts**: Critical warnings with visual and audio feedback
- 🎨 **Beautiful UI**: Modern, responsive Streamlit interface
- 🎥 **GoPro Integration**: Native support for GoPro Hero 8+ cameras

---

## ✨ Features

### 🎥 Camera System
- **Dual GoPro Support**: Front and rear camera monitoring
- **Multiple Connection Modes**: WiFi and USB connectivity
- **Remote Configuration**: Adjust resolution, FPS, and FOV
- **Live Streaming**: Real-time video feed processing
- **Frame Analysis**: 30-60 FPS detection capability

### 🤖 AI Detection
- **YOLO Object Detection**: Advanced neural network for accurate vehicle detection
- **Cascade Classifiers**: Fallback detection for reliability
- **Motion Detection**: Background subtraction for movement tracking
- **Multi-class Recognition**: Cars, trucks, buses, motorcycles, bicycles
- **Confidence Scoring**: Detection accuracy measurement

### 🚨 Alert System
- **Four Danger Levels**: Critical, High, Medium, Low
- **Real-time Notifications**: Instant visual alerts
- **Audio Warnings**: Sound-based danger detection
- **Position Tracking**: Front/rear/audio source identification
- **Alert History**: Last 3-5 alerts displayed

### 📊 Statistics Dashboard
- **Detection Counter**: Track threats by camera position
- **Connection Status**: Live GoPro connection monitoring
- **Frame Counter**: Performance monitoring
- **Session Statistics**: Total detections and alerts

### 🎨 User Interface
- **Modern Design**: Gradient-based, professional UI
- **Responsive Layout**: Adapts to different screen sizes
- **Live Video Feeds**: Side-by-side camera displays
- **Animated Alerts**: Pulsing danger notifications
- **Status Indicators**: Color-coded connection status

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GoPro CycleSafe AI                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Front GoPro │         │  Rear GoPro  │                  │
│  │   (WiFi/USB) │         │  (WiFi/USB)  │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         ├────────────────────────┤                           │
│         │                        │                           │
│  ┌──────▼────────────────────────▼───────┐                  │
│  │     Video Stream Processing            │                  │
│  │  • YOLO Object Detection               │                  │
│  │  • Motion Detection                    │                  │
│  │  • Distance Estimation                 │                  │
│  └──────┬─────────────────────────────────┘                  │
│         │                                                     │
│         │         ┌──────────────┐                           │
│         │         │  Microphone  │                           │
│         │         └──────┬───────┘                           │
│         │                │                                    │
│         │         ┌──────▼───────────┐                       │
│         │         │  Audio Analysis  │                       │
│         │         │  • FFT Analysis  │                       │
│         │         │  • Horn Detection│                       │
│         │         │  • Siren Detection│                      │
│         │         └──────┬───────────┘                       │
│         │                │                                    │
│  ┌──────▼────────────────▼───────┐                          │
│  │     Alert Generation           │                          │
│  │  • Danger Level Classification │                          │
│  │  • Multi-source Fusion         │                          │
│  └──────┬─────────────────────────┘                          │
│         │                                                     │
│  ┌──────▼─────────────────────────┐                          │
│  │   Streamlit Web Interface      │                          │
│  │  • Live Video Display          │                          │
│  │  • Real-time Alerts            │                          │
│  │  • Statistics Dashboard        │                          │
│  └────────────────────────────────┘                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Hardware**: Minimum 4GB RAM, 2GB free disk space
- **GoPro Cameras**: Hero 8, 9, 10, 11, 12, or newer

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/gopro-cyclesafe-ai.git
cd gopro-cyclesafe-ai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
opencv-python>=4.5.0
numpy>=1.21.0
sounddevice>=0.4.5
scipy>=1.7.0
requests>=2.26.0
```

### Step 4: Download YOLO Models (Optional - Recommended for Better Detection)

```bash
# Download YOLOv3-tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Download YOLOv3-tiny config
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg

# Download COCO names
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

Place these files in the project root directory.

---

## 🚀 Quick Start

### Option 1: Basic Setup (Any Camera)

```bash
streamlit run cycling_safety.py
```

This runs the basic version that works with any webcam or camera device.

### Option 2: GoPro Setup (Recommended)

```bash
streamlit run gopro_cycling_safety.py
```

This runs the full GoPro-optimized version with advanced features.

### Initial Configuration

1. **Open your browser** to `http://localhost:8501`
2. **Configure cameras** in the sidebar:
   - Enter GoPro IP addresses (default: 10.5.5.9, 10.5.5.10)
   - Or use camera device IDs (0, 1, 2, etc.)
3. **Adjust settings**:
   - Resolution: 1080p recommended
   - FPS: 30 for detection, 60 for smooth video
   - FOV: Linear recommended for minimal distortion
4. **Enable audio monitoring** (optional but recommended)
5. **Click "Start Monitoring"** to begin

---

## ⚙️ Configuration

### GoPro Setup

#### WiFi Connection Method

1. **Enable WiFi on GoPro**:
   - Swipe down → Preferences → Connections → WiFi → ON
   - Note the WiFi name and password

2. **Connect Computer to GoPro**:
   - Connect to GoPro WiFi network
   - Wait for connection to establish

3. **Find IP Address**:
   - Default IP: `10.5.5.9` (first camera)
   - Check GoPro Quik app for IP if needed
   - Second camera typically: `10.5.5.10`

4. **Enter in Application**:
   - Front GoPro IP: `10.5.5.9`
   - Rear GoPro IP: `10.5.5.10`

#### USB Connection Method (Recommended)

1. **Connect via USB-C**:
   - Use original GoPro USB-C cable
   - Connect to computer

2. **Enable Webcam Mode**:
   - GoPro Hero 9+: Settings → Connections → USB Connection → GoPro Webcam
   - GoPro Hero 8: Update firmware to enable webcam beta

3. **Automatic Detection**:
   - System will auto-detect USB cameras
   - No IP address needed

### Camera Settings

| Setting | Options | Recommended | Purpose |
|---------|---------|-------------|---------|
| Resolution | 480p, 720p, 1080p | **1080p** | Balance quality/performance |
| FPS | 30, 60, 120 | **30** | Detection accuracy |
| FOV | Wide, Medium, Narrow, Linear | **Linear** | Minimal distortion |

### Audio Settings

- **Sample Rate**: 44100 Hz (default)
- **Chunk Duration**: 1.5 seconds
- **Horn Frequency Range**: 300-700 Hz
- **Siren Frequency Range**: 800-1500 Hz
- **Loud Noise Threshold**: 0.25 RMS

---

## 📖 Usage Guide

### Starting a Session

1. **Prepare Hardware**:
   - Mount GoPros on bike (front handlebars, rear seat post)
   - Ensure stable mounting
   - Check battery levels
   - Enable WiFi or connect USB

2. **Launch Application**:
   ```bash
   streamlit run gopro_cycling_safety.py
   ```

3. **Configure & Connect**:
   - Enter camera settings
   - Click "Start Monitoring"
   - Wait for connection confirmation (green status)

4. **Begin Riding**:
   - Monitor alerts on screen/phone
   - Audio alerts will sound for critical dangers
   - System runs continuously

### Understanding Alerts

#### Danger Levels

| Level | Color | Description | Action |
|-------|-------|-------------|--------|
| 🚨 **CRITICAL** | Red | Vehicle very close (<3m) | IMMEDIATE ACTION |
| ⚠️ **HIGH** | Orange | Vehicle approaching (<5m) | Increase awareness |
| 🔶 **MEDIUM** | Light Orange | Vehicle nearby (<10m) | Monitor situation |
| ✅ **LOW** | Green | Vehicle distant (>10m) | Normal riding |

#### Alert Types

- **Vehicle Detected**: Car, truck, bus, or motorcycle identified
- **Horn Detected**: Vehicle horn sound recognized
- **Siren Detected**: Emergency vehicle approaching
- **Loud Noise**: Unusual loud sound detected
- **Motion Detected**: Significant movement in frame

### Dashboard Features

#### Live Video Feeds
- **Front Camera**: Shows forward-facing view with detection boxes
- **Rear Camera**: Shows rear view with approaching vehicles
- **Bounding Boxes**: Color-coded by danger level
- **Labels**: Object type and danger level displayed

#### Alert Panel
- **Latest Alert**: Large, animated display of most recent danger
- **Alert History**: Shows last 3-5 alerts with timestamps
- **Auto-clear**: Old alerts fade after new ones appear

#### Statistics Panel
- **Front Detections**: Count of threats from front
- **Rear Detections**: Count of threats from rear
- **Audio Alerts**: Count of sound-based warnings
- **Total**: Sum of all detections
- **Connection Status**: Real-time camera connectivity

---

## 🖥️ Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| CPU | Intel i5 / AMD Ryzen 5 or equivalent |
| RAM | 4GB |
| Storage | 2GB free space |
| GPU | Integrated graphics (Intel HD 4000+) |
| USB | USB 3.0 port (for USB connection) |
| WiFi | 802.11n (for WiFi connection) |

### Recommended Requirements

| Component | Specification |
|-----------|--------------|
| CPU | Intel i7 / AMD Ryzen 7 or better |
| RAM | 8GB+ |
| Storage | 5GB free space (for recordings) |
| GPU | Dedicated GPU (NVIDIA GTX 1050+) |
| USB | USB 3.1/3.2 ports |
| WiFi | 802.11ac or better |

### Supported GoPro Models

- ✅ GoPro Hero 12 Black
- ✅ GoPro Hero 11 Black / Mini
- ✅ GoPro Hero 10 Black
- ✅ GoPro Hero 9 Black
- ✅ GoPro Hero 8 Black (with firmware update)
- ⚠️ Hero 7 and older: Limited support via USB only

---

## 🔧 Troubleshooting

### Camera Connection Issues

**Problem**: "Cannot connect to GoPro"

**Solutions**:
1. Verify GoPro WiFi is enabled
2. Check computer is connected to GoPro network
3. Confirm IP address is correct (ping 10.5.5.9)
4. Try USB connection instead
5. Restart GoPro and reconnect

**Problem**: "Video feed frozen"

**Solutions**:
1. Check USB cable connection
2. Reduce resolution or FPS
3. Close other applications using camera
4. Restart the application

### Detection Issues

**Problem**: "No vehicles detected"

**Solutions**:
1. Ensure YOLO models are downloaded
2. Check lighting conditions (avoid direct sun)
3. Adjust camera FOV to Linear
4. Verify objects are in frame
5. Check if cascade classifier is working

**Problem**: "Too many false positives"

**Solutions**:
1. Increase detection confidence threshold
2. Adjust FOV to narrow field
3. Use Linear FOV to reduce distortion
4. Improve camera mounting stability

### Audio Issues

**Problem**: "No audio detection"

**Solutions**:
1. Check microphone permissions
2. Verify correct audio device selected
3. Test microphone in system settings
4. Increase volume threshold
5. Check if microphone is muted

### Performance Issues

**Problem**: "Lag or stuttering"

**Solutions**:
1. Reduce camera resolution to 720p
2. Lower FPS to 30
3. Close unnecessary applications
4. Disable YOLO (use cascade only)
5. Use USB instead of WiFi
6. Upgrade hardware (add GPU)

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: cv2` | OpenCV not installed | `pip install opencv-python` |
| `Can't open camera` | Camera in use | Close other apps using camera |
| `Connection timeout` | GoPro not responding | Check WiFi connection |
| `Audio device error` | Microphone unavailable | Check system permissions |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Reporting Bugs

1. Check existing issues
2. Create detailed bug report with:
   - System specifications
   - Steps to reproduce
   - Error messages
   - Screenshots/videos

### Suggesting Features

1. Open feature request issue
2. Describe use case
3. Explain expected behavior
4. Provide mockups if applicable

### Pull Requests

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with clear commits
4. Test thoroughly
5. Submit PR with description

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/gopro-cyclesafe-ai.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 *.py
black *.py
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 GoPro CycleSafe AI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **Streamlit** - Web application framework
- **YOLO** (Joseph Redmon) - Object detection algorithm
- **GoPro** - Camera hardware support
- **Python Community** - Various libraries and tools

---

## 🗺️ Roadmap

### Version 1.1 (Coming Soon)
- [ ] Mobile app for remote monitoring
- [ ] Cloud recording and storage
- [ ] GPS integration for route tracking
- [ ] Advanced analytics dashboard

### Version 2.0 (Future)
- [ ] Multi-cyclist group monitoring
- [ ] Machine learning model training
- [ ] Integration with bike computers
- [ ] Automated incident reporting

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ for safer cycling**

</div>
