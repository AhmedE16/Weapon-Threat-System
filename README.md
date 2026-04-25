# 🛡️ AI-Powered Weapon Threat & Loitering Detector

A real-time surveillance system built with **YOLOv8** and **Deep Re-ID** to detect weapons and monitor restricted areas.

## ✨ Key Features
* **Persistent Tracking:** Uses deep feature extraction to maintain object identity across frames.
* **Dynamic Spatial Zones:** Users can define restricted areas directly in the UI.
* **Automated Alerting:** Logs threats to CSV when a weapon loiters in a zone for >3s.
* **Rotation-Invariant Re-ID:** Identifies weapons even when held at different angles.

## 🚀 Installation
1. `git clone https://github.com/AhmedE16/Weapon-Detector.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`


## 📥 Model Download
Due to file size limits, the trained YOLOv8 weights are hosted on Google Drive.
1. Download `best_latest.pt` from [https://drive.google.com/drive/folders/1Xv9491xRwlLciHSd4ZDp3A8ThTdrXqr-?usp=sharing].
2. Place the file in the root directory of this project.
