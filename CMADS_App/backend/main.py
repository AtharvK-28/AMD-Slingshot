import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import os

# Apply nest_asyncio as requested
nest_asyncio.apply()

app = FastAPI(title="CMADS Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global YOLO model initialization
# Loading once when the app starts
try:
    model = YOLO("yolov8s.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Default video path - Update this to a valid path on your system
VIDEO_PATH = os.environ.get("VIDEO_PATH", r"C:\Users\Acer\Desktop\CrowdSenseAI\railway2.mp4")

# Optical Flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

@app.get("/")
async def root():
    return {"status": "online", "message": "CMADS API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Dashboard Client Connected via FastAPI WebSocket!")
    
    if model is None:
        await websocket.send_json({"error": "YOLO model not loaded"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error reading video at {VIDEO_PATH}. Please check the path.")
        await websocket.send_json({"error": f"Could not open video file at {VIDEO_PATH}"})
        await websocket.close()
        return

    try:
        ret, init_frame = cap.read()
        if not ret:
            await websocket.close()
            return
        
        h, w, _ = init_frame.shape
        mid_h, mid_w = h // 2, w // 2
        prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

        zone_coords = [
            (0, 0, mid_w, mid_h),           # Zone 1: Top-Left
            (mid_w, 0, w, mid_h),           # Zone 2: Top-Right
            (0, mid_h, mid_w, h),           # Zone 3: Bottom-Left
            (mid_w, mid_h, w, h)            # Zone 4: Bottom-Right
        ]

        grid_step = 30
        grid_y, grid_x = np.mgrid[grid_step/2:h:grid_step, grid_step/2:w:grid_step].reshape(2,-1).astype(int)
        base_points = np.vstack((grid_x, grid_y)).T.reshape(-1, 1, 2).astype(np.float32)

        platform_polygon = np.array([
            [int(w * 0.18), 0],    
            [w, 0],                
            [w, h],                
            [0, h]                 
        ], np.int32)
        
        base_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(base_mask, [platform_polygon], 255)

        cpi_history = deque(maxlen=30)
        frame_skip = 2
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            current_gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.bitwise_and(current_gray_raw, current_gray_raw, mask=base_mask)

            # YOLO Detection
            results = model(frame, classes=[0], conf=0.4, imgsz=640, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

            zone_people_counts = [0, 0, 0, 0]
            for box in boxes:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                if cy < mid_h and cx < mid_w: zone_people_counts[0] += 1
                elif cy < mid_h and cx >= mid_w: zone_people_counts[1] += 1
                elif cy >= mid_h and cx < mid_w: zone_people_counts[2] += 1
                else: zone_people_counts[3] += 1

            # Optical Flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, base_points, None, **LK_PARAMS)
            
            if next_points is not None and status is not None:
                good_new = next_points[status == 1]
                good_old = base_points[status == 1]
                vectors = good_new - good_old
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                magnitudes = np.linalg.norm(vectors, axis=1)

                motion_mask = magnitudes > 1.5
                active_points = good_old[motion_mask]
                active_angles = angles[motion_mask]
            else:
                active_points, active_angles = np.array([]), np.array([])

            cpi_values = []
            zones_payload = []

            for i, (x1, y1, x2, y2) in enumerate(zone_coords):
                zone_curr = current_gray[y1:y2, x1:x2]
                zone_prev = prev_gray[y1:y2, x1:x2]
                depth_weight = 1.8 if i < 2 else 1.0

                edges = cv2.Canny(zone_curr, 50, 150)
                raw_density = np.sum(edges / 255.0) / edges.size
                density = min((raw_density * 6.0) * depth_weight, 1.0)

                diff = cv2.absdiff(zone_prev, zone_curr)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                raw_motion = np.sum(thresh / 255.0) / thresh.size
                motion = min(raw_motion * 5.0, 1.0)

                normalized_count = min((zone_people_counts[i] * depth_weight) / 20.0, 1.0)

                chaos_multiplier = 1.0
                if len(active_points) > 0:
                    z_mask = (active_points[:, 0] >= x1) & (active_points[:, 0] < x2) & \
                             (active_points[:, 1] >= y1) & (active_points[:, 1] < y2)
                    zone_angles = active_angles[z_mask]
                    if len(zone_angles) > 5:
                        angle_std = np.std(zone_angles)
                        chaos_multiplier = 1.0 + min(angle_std / 3.0, 0.5)

                base_cpi = (0.5 * density) + (0.3 * normalized_count) + (0.2 * motion)
                cpi = min(base_cpi * chaos_multiplier, 1.0)
                cpi_values.append(cpi)

                status_text = "SAFE"
                if cpi >= 0.75: status_text = "CRITICAL"
                elif cpi >= 0.45: status_text = "WARNING"

                zones_payload.append({
                    "id": i + 1,
                    "name": f"Zone {i + 1}",
                    "cpi": round(float(cpi), 2),
                    "status": status_text,
                    "peopleCount": int(zone_people_counts[i])
                })

            global_risk = float(np.mean(cpi_values)) if cpi_values else 0.0
            cpi_history.append(global_risk)

            surge_warning = False
            if len(cpi_history) == cpi_history.maxlen:
                delta_cpi = global_risk - cpi_history[0]
                if delta_cpi > 0.15:
                    surge_warning = True

            # Encode frame to Base64
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "globalRisk": round(global_risk, 2),
                "surgeWarning": bool(surge_warning),
                "timestamp": int(time.time() * 1000),
                "zones": zones_payload,
                "frame": frame_base64
            }

            await websocket.send_json(payload)
            prev_gray = current_gray
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
