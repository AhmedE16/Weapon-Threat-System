import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time, os, csv, torch
from torchvision import models, transforms
from PIL import Image
from collections import Counter

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ── Sidebar Config ──────────────────────────────────────────────
st.sidebar.title("⚙️ System Settings")
CONF = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
LOITER_LIMIT = st.sidebar.slider("Loitering Time (sec)", 1.0, 10.0, 3.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Restricted Zone Setup")
# These inputs allow you to draw the box on the live feed
z_x1 = st.sidebar.number_input("Top Left X", 0, 1280, 100)
z_y1 = st.sidebar.number_input("Top Left Y", 0, 720, 100)
z_x2 = st.sidebar.number_input("Bottom Right X", 0, 1280, 500)
z_y2 = st.sidebar.number_input("Bottom Right Y", 0, 720, 450)
ZONE_XYXY = [z_x1, z_y1, z_x2, z_y2]

if st.sidebar.button("🗑️ Clear Threat Log"):
    with open("threat_logs.csv", 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "ID", "Class", "Duration", "Status"])
    st.sidebar.success("Log cleared!")

# ── Constants ───────────────────────────────────────────────────
MODEL_PATH   = "best_latest.pt"
DNA_THRESH_WEAPON = 0.45  
DNA_THRESH_KNIFE  = 0.58  
VOTING_WINDOW     = 25    
LOG_FILE = "threat_logs.csv"

CLASS_NAMES  = ["Sickle", "Machete", "Axe", "Sword", "Knife", "Pistol", "Rifle"]
COLORS       = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255),(255,165,0)]

# ── Load Models ─────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1]).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return extractor, preprocess, device

model = load_yolo()
extractor, preprocess, device = load_extractor()

# ── Session State (Memory Management) ───────────────────────────
if 'reid_memory' not in st.session_state: st.session_state.reid_memory = {}
if 'active_locks' not in st.session_state: st.session_state.active_locks = {}
if 'reid_voters' not in st.session_state: st.session_state.reid_voters = {}
if 'zone_timers' not in st.session_state: st.session_state.zone_timers = {}
if 'logged_threats' not in st.session_state: st.session_state.logged_threats = set()
if 'next_perm_id' not in st.session_state: st.session_state.next_perm_id = 1

# ── Tracking & Re-ID Logic (Restored) ───────────────────────────
def get_deep_dna(frame, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    norm_ratio = max(w, h) / min(w, h) if min(w, h) != 0 else 0
    nx1, ny1 = max(0, int(x1 + w*0.15)), max(0, int(y1 + h*0.15))
    nx2, ny2 = min(frame.shape[1], int(x2 - w*0.15)), min(frame.shape[0], int(y2 - h*0.15))
    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size < 50: return None, 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    crop = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = extractor(tensor).flatten()
    return (feature / feature.norm()).cpu().numpy(), norm_ratio

def process_identity(tracker_id, cls, dna, ratio):
    if tracker_id in st.session_state.active_locks:
        p_id = st.session_state.active_locks[tracker_id]
        gallery = st.session_state.reid_memory[p_id]["gallery"]
        if len(gallery) < 50:
            best_sim = max([np.dot(dna, v) for v in gallery])
            if best_sim < 0.90: gallery.append(dna)
        return p_id

    is_knife = (CLASS_NAMES[cls].lower() == "knife")
    current_thresh = DNA_THRESH_KNIFE if is_knife else DNA_THRESH_WEAPON
    best_candidate, max_score = None, 0
    
    for p_id, data in st.session_state.reid_memory.items():
        if p_id in st.session_state.active_locks.values() or data["class"] != cls: continue
        ratio_diff = abs(data["ratio"] - ratio)
        if is_knife and ratio_diff > 1.0: continue
        elif not is_knife and ratio_diff > 1.5: continue
        score = max([np.dot(dna, v) for v in data["gallery"]])
        if score > max_score:
            max_score = score
            best_candidate = p_id

    if tracker_id not in st.session_state.reid_voters: st.session_state.reid_voters[tracker_id] = []
    st.session_state.reid_voters[tracker_id].append(best_candidate if max_score > current_thresh else "NEW")

    if len(st.session_state.reid_voters[tracker_id]) >= VOTING_WINDOW:
        winner = Counter(st.session_state.reid_voters[tracker_id]).most_common(1)[0][0]
        if winner == "NEW":
            p_id = st.session_state.next_perm_id
            st.session_state.next_perm_id += 1
            st.session_state.reid_memory[p_id] = {"gallery": [dna], "ratio": ratio, "class": cls}
        else:
            p_id = winner
        st.session_state.active_locks[tracker_id] = p_id
        del st.session_state.reid_voters[tracker_id]
        return p_id
    return None

# ── Main UI Layout ──────────────────────────────────────────────
st.title("🛡️ Weapon Surveillance Dashboard")
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()
with col2:
    st.subheader("📊 Session Stats")
    stats_placeholder = st.empty()
    st.subheader("🚨 Real-Time Logs")
    log_placeholder = st.empty()

# ── Execution Loop ──────────────────────────────────────────────
run_detection = st.sidebar.checkbox("▶️ Start Live Stream", value=False)

if run_detection:
    cap = cv2.VideoCapture(0)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "ID", "Class", "Duration", "Status"])
    
    while run_detection:
        ret, frame = cap.read()
        if not ret: break
        
        # Draw the Dynamic Zone
        cv2.rectangle(frame, (ZONE_XYXY[0], ZONE_XYXY[1]), (ZONE_XYXY[2], ZONE_XYXY[3]), (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED AREA", (ZONE_XYXY[0], ZONE_XYXY[1]-10), 1, 1, (0, 0, 255), 2)

        results = model.track(frame, conf=CONF, persist=True, tracker="botsort.yaml", verbose=False)[0]
        active_tids = set()
        threat_count = 0
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                active_tids.add(tid)
                
                dna, ratio = get_deep_dna(frame, x1, y1, x2, y2)
                if dna is None: continue
                
                perm_id = process_identity(tid, cls, dna, ratio)
                
                # Logic for Restricted Area
                status_txt = "MONITORING"
                color = COLORS[cls % len(COLORS)]
                
                if perm_id:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    is_inside = (ZONE_XYXY[0] < cx < ZONE_XYXY[2]) and (ZONE_XYXY[1] < cy < ZONE_XYXY[3])
                    
                    if is_inside:
                        if perm_id not in st.session_state.zone_timers:
                            st.session_state.zone_timers[perm_id] = time.time()
                        
                        duration = time.time() - st.session_state.zone_timers[perm_id]
                        if duration > LOITER_LIMIT:
                            status_txt, color = "!!! THREAT !!!", (0, 0, 255)
                            threat_count += 1
                            if perm_id not in st.session_state.logged_threats:
                                with open(LOG_FILE, 'a', newline='') as f:
                                    csv.writer(f).writerow([time.strftime("%H:%M:%S"), perm_id, CLASS_NAMES[cls], round(duration, 1), "ALARM"])
                                st.session_state.logged_threats.add(perm_id)
                        else:
                            status_txt = f"ZONE ENTRY: {round(duration, 1)}s"
                    else:
                        st.session_state.zone_timers.pop(perm_id, None)
                        st.session_state.logged_threats.discard(perm_id)

                # Rendering Detecton
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{perm_id} {CLASS_NAMES[cls]}" if perm_id else "IDENTIFYING..."
                cv2.putText(frame, label, (x1, y1-10), 1, 0.8, color, 2)
                if perm_id:
                    cv2.putText(frame, status_txt, (x1, y2+25), 1, 0.7, color, 2)

        # Sync memory
        for tid in list(st.session_state.active_locks.keys()):
            if tid not in active_tids: del st.session_state.active_locks[tid]
        for tid in list(st.session_state.reid_voters.keys()):
            if tid not in active_tids: del st.session_state.reid_voters[tid]
        
        # Display Frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Update Stats Sidebar/Column
        with stats_placeholder.container():
            st.write(f"**Target Threats:** {threat_count}")
            st.write(f"**Objects in Memory:** {len(st.session_state.reid_memory)}")
        
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            log_placeholder.dataframe(df.tail(8), use_container_width=True)

    cap.release()
else:
    st.info("Check the 'Start Live Stream' box to activate camera.")