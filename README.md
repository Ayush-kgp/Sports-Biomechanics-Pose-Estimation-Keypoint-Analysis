
# Pose-based Sports Analytics on a Cricket Video

## Overview
This project is a small computer vision pipeline for cricket movement analysis.  
It takes a side-view cricket video, runs pose estimation, stores the extracted keypoints frame-by-frame, and then computes simple but useful movement metrics such as joint angles and stability.

---

## Video Used
I used a **side-view cricket action clip**, since side angles make it much easier to observe posture changes.  
From this view, knee bend and torso tilt are clearly visible and less affected by perspective distortion compared to front or diagonal angles.

---

## Pose Model
For pose estimation, I used **YOLOv8 Pose (Ultralytics)** with the pretrained model `yolov8n-pose.pt`.

I chose YOLOv8 Pose because it is easy to run on CPU, provides stable keypoints in real-world videos, and also returns confidence scores which are useful for filtering noisy detections.

---

## Outputs
After running the script, the results are saved inside `outputs/`:

- `overlay.mp4` — video with pose skeleton drawn on each frame  
- `keypoints.csv` — extracted keypoints (x, y + confidence) stored frame-wise  
- `metrics.csv` — computed movement metrics  
- `metrics.png` — plot of the metrics over time  

---

## Metrics Extracted
The following pose-based metrics were computed:

**1) Knee angle (deg)**  
Computed using hip → knee → ankle. This helps capture knee flexion and lower-body stability during the action.

**2) Torso lean (deg)**  
Measured using the hip-midpoint → shoulder-midpoint direction relative to vertical. This reflects posture and body balance.

**3) Jitter / stability (px)**  
Frame-to-frame movement of a tracked point (nose, and shoulder midpoint if nose is unreliable). This helps quantify pose noise and sudden tracking jumps, especially during fast movement or occlusion.
## Why these metrics are relevant to cricket
- **Knee angle**: shows leg stability and flexion during shot execution and balance control which is important for strong base and injury prevention.
- **Torso lean**: captures posture and weight transfer, which affects shot timing, power generation, and body alignment.
- **Jitter/stability**: indicates tracking stability and also reflects sudden movements/occlusions common in cricket actions like bat swing, quick footwork etc.

---

## Future Work

### What problems did I notice?

- Occlusion & Crossing: When the bat or arms cross the torso during a follow-through, the model occasionally "swaps" joints or loses tracking. 
- Motion Blur Jitter: High-velocity movements (like the foot plant or wrist snap) create blur that causes the keypoints—especially the ankles—to "jump" between frames. 

- Dimensionality: Working with a single 2D camera means we are calculating "projected" angles. For example, a knee bend might look sharper or flatter depending on how much the player is angled toward the lens.

---

### How can this be improved?
If I had more time/data, I would improve reliability like this:
- Instead of treating each frame as a standalone image, I'd implement a Kalman Filter or a One-Euro Filter. These are much better than a simple moving average because they account for velocity, making the skeleton feel "heavy" and less prone to erratic jitter. 

- I would also use the Confidence Scores returned by the model as a gatekeeper. If the model is only 30% sure about a foot position, we should interpolate that position from the surrounding frames rather than plotting a "ghost" point.
---

### What kind of data would I collect?
To make the system work in real conditions, I’d collect:
- different camera angles (side view, diagonal, front)
- different lighting (day, night, indoor nets)
- different backgrounds (stadium, practice nets, crowd)
- different actions (batting, bowling, sprinting, fielding dives)
- lots of examples where the bat/arms block joints (real occlusion cases)

---

### How should the dataset be split?
I would not split randomly by frames, because frames from the same video are very similar.
Instead, I would split by player and by full video, like 70% train, 15% validation and 15% test

This is a better way of generalisation.

---

### How would I evaluate it?
- I would evaluate the next iteration using: PCK (Percentage of Correct Keypoints): To see if our "estimated" joints fall within a normalized distance of the true joint. 

- Jitter Variance: Measuring the "noise" in the angle signal over time; a lower variance in a stationary pose indicates a more stable model.

## How to Run
```bash
pip install -r requirements.txt
python main.py

---
output videp : https://drive.google.com/file/d/1musrX-uroWwQm9oz0kVueRfN3rOKkt47/view?usp=sharing
