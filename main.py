import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
INPUT_VIDEO = "input_video.mov"   # update if needed
OUT_DIR = "outputs"

OVERLAY_VIDEO = os.path.join(OUT_DIR, "overlay.mp4")
KEYPOINTS_CSV = os.path.join(OUT_DIR, "keypoints.csv")
METRICS_CSV = os.path.join(OUT_DIR, "metrics.csv")
METRICS_PNG = os.path.join(OUT_DIR, "metrics.png")

os.makedirs(OUT_DIR, exist_ok=True)

# YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")  # auto-downloads weights

# -----------------------------
# Helpers
# -----------------------------
def angle_3pts(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def moving_average(x, w=9):
    x = np.array(x, dtype=np.float32)
    if len(x) == 0:
        return x
    w = max(1, int(w))
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xpad, kernel, mode="valid")

# COCO keypoint indices for YOLO Pose (17 keypoints)
# 0 nose, 5 left_shoulder, 6 right_shoulder, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
KP = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

def safe_midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

def get_pt(kpts, idx, conf_th=0.3):
    """
    kpts: shape (17,3) => x,y,conf
    """
    x, y, c = kpts[idx]
    if c < conf_th:
        return None
    return (float(x), float(y))

# -----------------------------
# Video IO
# -----------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OVERLAY_VIDEO, fourcc, fps, (W, H))

records = []
missed = 0

print(f"[INFO] Running YOLOv8 Pose on {nframes} frames @ {fps:.2f} FPS")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # predict
    results = model.predict(frame, verbose=False)
    r = results[0]

    row = {"frame": frame_idx, "time_s": frame_idx / fps}

    if r.keypoints is None or len(r.keypoints) == 0:
        missed += 1
        # save NaNs for 17 keypoints
        for name, idx in KP.items():
            row[f"{name}_x"] = np.nan
            row[f"{name}_y"] = np.nan
            row[f"{name}_c"] = np.nan
    else:
        # choose person with highest confidence (best detection)
        # keypoints.data shape: (num_people, 17, 3)
        kpts_all = r.keypoints.data.cpu().numpy()
        # choose person by max mean conf
        confs = kpts_all[:, :, 2].mean(axis=1)
        best = int(np.argmax(confs))
        kpts = kpts_all[best]

        # draw skeleton overlay using YOLO built-in plotting
        annotated = r.plot()
        frame = annotated

        # store keypoints
        for name, idx in KP.items():
            row[f"{name}_x"] = float(kpts[idx, 0])
            row[f"{name}_y"] = float(kpts[idx, 1])
            row[f"{name}_c"] = float(kpts[idx, 2])

    writer.write(frame)
    records.append(row)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f" processed {frame_idx}/{nframes}")

cap.release()
writer.release()

df = pd.DataFrame(records)
df.to_csv(KEYPOINTS_CSV, index=False)

print(f"[DONE] overlay video: {OVERLAY_VIDEO}")
print(f"[DONE] keypoints csv: {KEYPOINTS_CSV}")
print(f"[INFO] missed frames: {missed}/{frame_idx} = {missed/max(1,frame_idx)*100:.2f}%")

# -----------------------------
# Metrics
# -----------------------------
metrics = []

prev_track_pt = None
jitter_vals = []

for _, r in df.iterrows():
    # Extract points (in pixel coords)
    nose = None if np.isnan(r["nose_x"]) else (r["nose_x"], r["nose_y"])

    lhip  = None if np.isnan(r["l_hip_x"]) else (r["l_hip_x"], r["l_hip_y"])
    lknee = None if np.isnan(r["l_knee_x"]) else (r["l_knee_x"], r["l_knee_y"])
    lank  = None if np.isnan(r["l_ankle_x"]) else (r["l_ankle_x"], r["l_ankle_y"])

    lsh   = None if np.isnan(r["l_shoulder_x"]) else (r["l_shoulder_x"], r["l_shoulder_y"])
    rsh   = None if np.isnan(r["r_shoulder_x"]) else (r["r_shoulder_x"], r["r_shoulder_y"])
    rhip  = None if np.isnan(r["r_hip_x"]) else (r["r_hip_x"], r["r_hip_y"])

    shoulder_mid = safe_midpoint(lsh, rsh)
    hip_mid = safe_midpoint(lhip, rhip)

    # Metric 1: left knee angle
    if lhip is None or lknee is None or lank is None:
        knee_angle = np.nan
    else:
        knee_angle = angle_3pts(lhip, lknee, lank)

    # Metric 2: torso lean angle from vertical
    if shoulder_mid is None or hip_mid is None:
        torso_lean = np.nan
    else:
        vx = shoulder_mid[0] - hip_mid[0]
        vy = shoulder_mid[1] - hip_mid[1]
        denom = (np.sqrt(vx*vx + vy*vy) * 1.0) + 1e-9
        # vertical axis (0, -1)
        cosang = (vx*0 + vy*(-1)) / denom
        cosang = np.clip(cosang, -1, 1)
        torso_lean = float(np.degrees(np.arccos(cosang)))

    # Metric 3: jitter (nose movement per frame)
    track_pt = nose if nose is not None else shoulder_mid

    if track_pt is None:
        jitter = np.nan
        prev_track_pt = None
    else:
        if prev_track_pt is None:
            jitter = 0.0
        else:
            dx = track_pt[0] - prev_track_pt[0]
            dy = track_pt[1] - prev_track_pt[1]
            jitter = float(np.sqrt(dx*dx + dy*dy))
        prev_track_pt = track_pt

    jitter_vals.append(jitter)

    metrics.append({
        "frame": r["frame"],
        "time_s": r["time_s"],
        "knee_angle_deg": knee_angle,
        "torso_lean_deg": torso_lean,
        "jitter_px": jitter,
    })

m = pd.DataFrame(metrics)

# smoothing
m["knee_angle_smooth"] = moving_average(
    m["knee_angle_deg"].ffill().bfill().values, 9
)

m["torso_lean_smooth"] = moving_average(
    m["torso_lean_deg"].ffill().bfill().values, 9
)
m["jitter_smooth"] = moving_average(m["jitter_px"].fillna(0).values, 9)

m.to_csv(METRICS_CSV, index=False)
print(f"[DONE] metrics saved: {METRICS_CSV}")

# plot
plt.figure(figsize=(10, 6))
plt.plot(m["time_s"], m["knee_angle_smooth"], label="Knee angle (deg)")
plt.plot(m["time_s"], m["torso_lean_smooth"], label="Torso lean (deg)")
plt.plot(m["time_s"], m["jitter_smooth"], label="Jitter (px)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Pose-based movement metrics (YOLOv8 Pose)")
plt.legend()
plt.tight_layout()
plt.savefig(METRICS_PNG, dpi=200)
plt.close()
print(f"[DONE] plot saved: {METRICS_PNG}")
