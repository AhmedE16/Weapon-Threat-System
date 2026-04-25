# 🛡️ AI-Powered Weapon Threat & Loitering Detector

A real-time surveillance system built with **YOLOv8** and **Deep Re-ID** to detect weapons and monitor restricted areas.

## ✨ Key Features
* **Persistent Tracking:** Uses deep feature extraction to maintain object identity across frames.
* **Dynamic Spatial Zones:** Users can define restricted areas directly in the UI.
* **Automated Alerting:** Logs threats to CSV when a weapon loiters in a zone for >3s.
* **Rotation-Invariant Re-ID:** Identifies weapons even when held at different angles.

## 🚀 Installation
1. `git clone https://github.com/YourUser/Weapon-Detector.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
