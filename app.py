'''import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def run_traffic_system(
    lane_paths,
    yolo_weight,
    output_path,
    conf=0.4,
    iou=0.5,
    progress_callback=None
):
    # ================= CONFIG =================
    BASE_GREEN = 8
    SCALE_FACTOR = 1.5
    MIN_GREEN = 6
    MAX_GREEN = 30

    EMERGENCY_CLASS_IDS = {0, 4}  # ambulance, firetruck

    # ================= DIRECTORIES =================
    os.makedirs("violations", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ================= LOAD VIDEOS =================
    caps, fps_list, sizes = [], [], []

    for path in lane_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        caps.append(cap)
        fps_list.append(cap.get(cv2.CAP_PROP_FPS) or 30)
        sizes.append((
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))

    num_lanes = len(caps)
    fps = int(min(fps_list))
    W = max(w for w, h in sizes)
    H = max(h for w, h in sizes)

    # ================= OUTPUT GRID =================
    cols = 2
    rows = (num_lanes + 1) // 2
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (W * cols, H * rows)
    )

    # ================= YOLO =================
    models = [YOLO(yolo_weight) for _ in range(num_lanes)]
    lane_names = [f"lane{i+1}" for i in range(num_lanes)]

    stop_line_y = int(H * 0.6)
    last_centroids = {ln: {} for ln in lane_names}

    # ================= SIGNAL STATE =================
    active_green = lane_names[0]
    green_start_frame = 0
    green_duration = BASE_GREEN
    emergency_active = False

    # ================= STATS =================
    stats = {
        "active_green": active_green,
        "remaining_time": green_duration,
        "lane_counts": {ln: 0 for ln in lane_names},
        "violations": 0,
        "emergencies": 0
    }

    violated_ids = set()
    violation_log = []
    frame_idx = 0

    # ================= MAIN LOOP =================
    while True:
        frame_idx += 1
        frames = {}
        counts = {ln: set() for ln in lane_names}
        emergency_lane = None
        any_frame = False

        for i, cap in enumerate(caps):
            ln = lane_names[i]
            ret, frame = cap.read()

            if not ret:
                frames[ln] = np.zeros((H, W, 3), dtype=np.uint8)
                continue

            any_frame = True
            frame = cv2.resize(frame, (W, H))

            res = models[i].track(
                frame,
                persist=True,
                conf=conf,
                iou=iou,
                tracker="bytetrack.yaml",
                verbose=False
            )

            if res and res[0].boxes is not None:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)
                ids = (
                    res[0].boxes.id.cpu().numpy().astype(int)
                    if res[0].boxes.id is not None
                    else list(range(len(boxes)))
                )

                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cy = (y1 + y2) // 2
                    tid = f"{ln}_{ids[j]}"
                    cls_id = cls_ids[j]
                    cls_name = models[i].names[cls_id]

                    counts[ln].add(tid)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    last_y = last_centroids[ln].get(tid)
                    last_centroids[ln][tid] = cy

                    if cls_id in EMERGENCY_CLASS_IDS:
                        emergency_lane = ln

                    if (
                        last_y is not None and
                        last_y < stop_line_y <= cy and
                        ln != active_green and
                        cls_id not in EMERGENCY_CLASS_IDS
                    ):
                        if tid not in violated_ids:
                            violated_ids.add(tid)
                            stats["violations"] += 1

                            violation_log.append({
                                "frame": frame_idx,
                                "lane": ln,
                                "track_id": tid,
                                "class": cls_name,
                                "timestamp_sec": round(frame_idx / fps, 2)
                            })

                            cv2.imwrite(
                                f"violations/{ln}_frame_{frame_idx}_id_{ids[j]}.jpg",
                                frame
                            )

            frames[ln] = frame

        if not any_frame:
            break

        if emergency_lane and not emergency_active:
            emergency_active = True
            active_green = emergency_lane
            green_start_frame = frame_idx
            green_duration = MAX_GREEN
            stats["emergencies"] += 1

        elapsed = (frame_idx - green_start_frame) / fps
        if elapsed >= green_duration:
            emergency_active = False
            active_green = max(counts, key=lambda k: len(counts[k]))
            green_duration = min(
                MAX_GREEN,
                max(MIN_GREEN, int(BASE_GREEN + len(counts[active_green]) * SCALE_FACTOR))
            )
            green_start_frame = frame_idx

        stats["active_green"] = active_green
        stats["remaining_time"] = max(0, int(green_duration - elapsed))
        stats["lane_counts"] = {k: len(v) for k, v in counts.items()}

        grid = []
        for ln in lane_names:
            f = frames[ln]
            color = (0, 255, 0) if ln == active_green else (0, 0, 255)
            label = "GREEN" if ln == active_green else "RED"
            cv2.putText(f, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            grid.append(f)

        while len(grid) < rows * cols:
            grid.append(np.zeros((H, W, 3), dtype=np.uint8))

        final = cv2.vconcat(
            [cv2.hconcat(grid[r * cols:(r + 1) * cols]) for r in range(rows)]
        )

        out.write(final)

        if progress_callback:
            progress_callback(frame_idx, final, stats)

    for cap in caps:
        cap.release()
    out.release()

    # ================= SAVE VIOLATION EXCEL =================
    if violation_log:
        df = pd.DataFrame(violation_log)
        excel_path = output_path.replace(".avi", "_violations.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"[INFO] Violation log saved to {excel_path}")

    return output_path'''


import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime


def run_traffic_system(
    lane_paths,
    yolo_weight,
    output_path,
    conf=0.4,
    iou=0.5,
    progress_callback=None
):

    # ================= CONFIG =================
    BASE_GREEN = 8
    SCALE_FACTOR = 1.5
    MIN_GREEN = 6
    MAX_GREEN = 30

    EMERGENCY_CLASS_IDS = {0, 4}  # ambulance, firetruck

    # ================= SPEED OPTIMIZATION =================
    frame_skip = 3
    PROCESS_W = 640
    PROCESS_H = 360

    # ================= DIRECTORIES =================
    os.makedirs("violations", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ================= LOAD VIDEOS =================
    caps, fps_list = [], []

    for path in lane_paths:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        caps.append(cap)
        fps_list.append(cap.get(cv2.CAP_PROP_FPS) or 30)

    num_lanes = len(caps)
    fps = int(min(fps_list))

    W = PROCESS_W
    H = PROCESS_H

    # ================= OUTPUT GRID =================
    cols = 2
    rows = (num_lanes + 1) // 2

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (W * cols, H * rows)
    )

    # ================= YOLO =================
    model = YOLO(yolo_weight)

    lane_names = [f"lane{i+1}" for i in range(num_lanes)]

    stop_line_y = int(H * 0.6)

    last_centroids = {ln: {} for ln in lane_names}

    # ================= SIGNAL STATE =================
    active_green = lane_names[0]
    green_start_frame = 0
    green_duration = BASE_GREEN
    emergency_active = False

    # 🔥 NEW: Rotation Logic
    rotation_order = lane_names.copy()
    rotation_index = 0

    def update_rotation_order(counts):
        densities = [(ln, len(counts[ln])) for ln in lane_names]
        densities.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in densities]

    # ================= STATS =================
    stats = {
        "active_green": active_green,
        "remaining_time": green_duration,
        "lane_counts": {ln: 0 for ln in lane_names},
        "violations": 0,
        "emergencies": 0
    }

    violated_ids = set()
    violation_log = []

    frame_idx = 0

    # ================= MAIN LOOP =================
    while True:

        frame_idx += 1

        frames = {}
        counts = {ln: set() for ln in lane_names}

        emergency_lane = None
        any_frame = False

        for i, cap in enumerate(caps):

            ln = lane_names[i]

            ret, frame = cap.read()

            if not ret:
                frames[ln] = np.zeros((H, W, 3), dtype=np.uint8)
                continue

            any_frame = True

            frame = cv2.resize(frame, (W, H))

            if frame_idx % frame_skip != 0:
                frames[ln] = frame
                continue

            res = model.track(
                frame,
                persist=True,
                conf=conf,
                iou=iou,
                tracker="bytetrack.yaml",
                verbose=False
            )

            if res and res[0].boxes is not None:

                boxes = res[0].boxes.xyxy.cpu().numpy()
                cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)

                ids = (
                    res[0].boxes.id.cpu().numpy().astype(int)
                    if res[0].boxes.id is not None
                    else list(range(len(boxes)))
                )

                for j, box in enumerate(boxes):

                    x1, y1, x2, y2 = map(int, box)
                    cy = (y1 + y2) // 2

                    tid = f"{ln}_{ids[j]}"
                    cls_id = cls_ids[j]
                    cls_name = model.names[cls_id]

                    counts[ln].add(tid)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    last_y = last_centroids[ln].get(tid)
                    last_centroids[ln][tid] = cy

                    if cls_id in EMERGENCY_CLASS_IDS:
                        emergency_lane = ln

                    if (
                        last_y is not None
                        and last_y < stop_line_y <= cy
                        and ln != active_green
                        and cls_id not in EMERGENCY_CLASS_IDS
                    ):

                        if tid not in violated_ids:

                            violated_ids.add(tid)
                            stats["violations"] += 1

                            now = datetime.now()

                            image_filename = f"{ln}_frame_{frame_idx}_id_{ids[j]}.jpg"

                            image_path = os.path.abspath(os.path.join(
                                "violations", image_filename
                            ))

                            violation_log.append({
                                "date": now.strftime("%Y-%m-%d"),
                                "time": now.strftime("%H:%M:%S"),
                                "frame": frame_idx,
                                "lane": ln,
                                "track_id": tid,
                                "class": cls_name,
                                "timestamp_sec": round(frame_idx / fps, 2),
                                "image_file": image_filename,
                                "image_link": f'=HYPERLINK("{image_path}", "Open Image")'
                            })

                          

                            df.to_excel(excel_path, index=False, engine="xlsxwriter")

                            violation_img = frame.copy()

                            cv2.rectangle(
                                violation_img,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                3
                            )

                            cv2.putText(
                                violation_img,
                                "VIOLATION",
                                (x1, max(y1 - 10, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3
                            )

                            cv2.imwrite(image_path, violation_img)

            frames[ln] = frame

        if not any_frame:
            break

        # ---------- EMERGENCY ----------
        if emergency_lane and not emergency_active:

            emergency_active = True
            active_green = emergency_lane
            green_start_frame = frame_idx
            green_duration = MAX_GREEN
            stats["emergencies"] += 1

        # ---------- NORMAL ----------
        elapsed = (frame_idx - green_start_frame) / fps

        if elapsed >= green_duration:

            emergency_active = False

            # 🔥 Rotation-based selection
            rotation_index += 1

            if rotation_index >= len(rotation_order):
                rotation_index = 0
                rotation_order = update_rotation_order(counts)

            if rotation_index == 0:
                rotation_order = update_rotation_order(counts)

            active_green = rotation_order[rotation_index]

            green_duration = min(
                MAX_GREEN,
                max(
                    MIN_GREEN,
                    int(BASE_GREEN + len(counts[active_green]) * SCALE_FACTOR)
                )
            )

            green_start_frame = frame_idx

        # ---------- STATS ----------
        stats["active_green"] = active_green
        stats["remaining_time"] = max(0, int(green_duration - elapsed))
        stats["lane_counts"] = {k: len(v) for k, v in counts.items()}

        # ---------- GRID ----------
        grid = []

        for ln in lane_names:

            f = frames[ln]

            center = (25, 25)
            radius = 18

            color = (0, 255, 0) if ln == active_green else (0, 0, 255)

            cv2.circle(f, center, radius, color, -1)

            grid.append(f)

        while len(grid) < rows * cols:
            grid.append(np.zeros((H, W, 3), dtype=np.uint8))

        final = cv2.vconcat(
            [cv2.hconcat(grid[r * cols:(r + 1) * cols]) for r in range(rows)]
        )

        out.write(final)

        if progress_callback:
            progress_callback(frame_idx, final, stats)

    for cap in caps:
        cap.release()

    out.release()

    if len(violation_log) > 0:

        df = pd.DataFrame(violation_log)

        excel_path = os.path.join(
            os.path.abspath("violations"),
            f"violation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        df.to_excel(excel_path, index=False)

        print(f"[INFO] Violation log saved to: {excel_path}")

    else:
        print("[INFO] No violations detected")

    # Excel path (latest one)
    excel_path = None
    
    if len(violation_log) > 0:
        excel_path = os.path.join(
            os.path.abspath("violations"),
            f"violation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    
    # Return everything
    return output_path, excel_path, os.path.abspath("violations")
'''import os
import time
import streamlit as st

from backend.traffic_engine import run_traffic_system


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Based Traffic Signal Control",
    layout="wide"
)

st.title("🚦 AI-Based Traffic Signal Control")

st.markdown(
    "Upload lane videos. The system will process frames and stream output live."
)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload lane videos",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True
)

# -------------------------------------------------
# RUN BUTTON
# -------------------------------------------------
if st.button("Run System"):

    if not uploaded_files:
        st.error("Please upload at least one video.")
        st.stop()

    # ---------------- DIRECTORIES ----------------
    os.makedirs("videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ---------------- SAVE FILES ----------------
    lane_paths = []
    for i, file in enumerate(uploaded_files):
        path = f"videos/lane{i+1}.mp4"
        with open(path, "wb") as f:
            f.write(file.read())
        lane_paths.append(path)

    # ---------------- UI ELEMENTS ----------------
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    live_frame = st.empty()   # 🔴 LIVE STREAM CONTAINER

    MAX_FRAMES = 300

    # ---------------- CALLBACK ----------------
    def update_progress(frame_idx, frame=None):
        progress_bar.progress(min(frame_idx / MAX_FRAMES, 1.0))
        status_text.text(f"Processing frame {frame_idx}/{MAX_FRAMES}")

        if frame is not None:
            live_frame.image(
                frame,
                channels="BGR",
                use_column_width=True
            )

    # ---------------- RUN BACKEND ----------------
    with st.spinner("Processing video stream..."):
        output_path = run_traffic_system(
            lane_paths=lane_paths,
            yolo_weight="models/yolo11s.pt",
            output_path="outputs/output.avi",
            progress_callback=update_progress
        )

    time.sleep(1)

    # ---------------- FINAL OUTPUT ----------------
    if os.path.exists(output_path):
        st.success("✅ Processing completed successfully!")

        st.download_button(
            label="⬇ Download Output Video",
            data=open(output_path, "rb"),
            file_name="traffic_output.avi",
            mime="video/avi"
        )
    else:
        st.error("❌ Output video not found.")
'''

import os
import streamlit as st
from backend.traffic_engine import run_traffic_system

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙️ Configuration")

uploaded_files = st.sidebar.file_uploader(
    "Upload lane videos",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True
)

model_path = st.sidebar.text_input(
    "YOLOv11 weights path",
    value="model/best.pt"
)

start_btn = st.sidebar.button("▶ Start System")
stop_btn = st.sidebar.button("⏹ Stop System")

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.markdown("## Yolo Based ATS Control System With Real Time Violation Monitoring")

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
_, center_col, right_col = st.columns([0.2, 6.5, 3.3])

with center_col:
    st.markdown("### 📹 Live feed")
    live_frame = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

with right_col:
    st.markdown("### 🎛 Controls & status")
    system_state = st.empty()
    st.markdown("---")

    st.markdown("**Active green**")
    active_green_box = st.empty()

    st.markdown("**Remaining (s)**")
    remaining_time_box = st.empty()

    st.markdown("**Lane counts**")
    lane_count_box = st.empty()

    st.markdown("**Violations / Emergencies**")
    violation_box = st.empty()

# -------------------------------------------------
# RUN SYSTEM
# -------------------------------------------------
MAX_FRAMES = 3

if start_btn:

    if not uploaded_files:
        st.sidebar.error("Please upload at least one video.")
        st.stop()

    os.makedirs("videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    lane_paths = []
    for i, f in enumerate(uploaded_files):
        path = f"videos/lane{i+1}.mp4"
        with open(path, "wb") as out:
            out.write(f.read())
        lane_paths.append(path)

    system_state.success("🟢 Controller running")

    def update_progress(frame_idx, frame=None, stats=None):
        progress_bar.progress(min(frame_idx / MAX_FRAMES, 1.0))
        status_text.text(f"Processing frame {frame_idx}/{MAX_FRAMES}")

        if frame is not None:
            live_frame.image(frame, channels="BGR", width=900)

        if stats:
            active_green_box.markdown(f"### {stats['active_green']}")
            remaining_time_box.markdown(f"### {stats['remaining_time']}")
            lane_count_box.json(stats["lane_counts"])
            violation_box.markdown(
                f"🚨 **Violations:** {stats['violations']} &nbsp;&nbsp; "
                f"🚑 **Emergencies:** {stats['emergencies']}",
                unsafe_allow_html=True
            )

    output_path, excel_path, violation_dir = run_traffic_system(
        lane_paths=lane_paths,
        yolo_weight=model_path,
        output_path="outputs/output.avi",
        progress_callback=update_progress
    )

    system_state.success("🟢 Controller stopped")

    if os.path.exists(output_path):
    
        # 🎥 Video
        st.download_button(
            "⬇ Download Output Video",
            data=open(output_path, "rb"),
            file_name="traffic_output.avi"
        )
    
        # 📊 Excel
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    "📊 Download Violation Report",
                    data=f,
                    file_name="violation_report.xlsx"
                )
    
        # 📸 Images ZIP
        import zipfile
    
        zip_path = "violations.zip"
    
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(violation_dir):
                file_path = os.path.join(violation_dir, file)
                zipf.write(file_path, file)
    
        if os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    "📸 Download Violation Images",
                    data=f,
                    file_name="violations.zip"
                )

if stop_btn:
    system_state.warning("🛑 Controller stopped by user")

'''from ultralytics import YOLO
model = YOLO("Model/best.pt")
print(model.names)'''
