🏙️ CrowdSense AI
Predictive Crowd Management & Decision Support System
An official submission for the AMD Slingshot Hackathon

(Replace the path above with your Red Zone surge screenshot in your repo)

CrowdSense AI is a cutting-edge, privacy-first computer vision dashboard designed to shift public safety from reactive surveillance to proactive decision support. By analyzing crowd fluid dynamics (density, motion variance, and directional vectors) rather than tracking biometric identities, the system predicts crush events and stampedes before they form, and actively routes crowds to safe exits.

🚀 Features
The Crowd Pressure Index (CPI): A custom, depth-weighted metric combining Canny edge density, absolute frame differencing, and YOLO bounding boxes to determine true spatial pressure.

Temporal Surge Detection (Δ CPI): A rolling memory queue calculates the rate of crowd accumulation, triggering an immediate SURGE WARNING if a rapid influx occurs in under 3 seconds.

Chaos Multiplier: Utilizes grid-based Lucas-Kanade Optical Flow to calculate the standard deviation of movement angles. If uniform flow scatters into multi-directional panic, the risk score instantly multiplies.

Dynamic A* Escape Routing: The system doesn't just panic; it acts. Upon detecting a Red Choke Point, the integrated A* pathfinding engine automatically recalculates and highlights the safest alternative exit route for authorities.

Privacy by Design: Zero facial recognition. The system tracks pixels, vectors, and bounding boxes, ensuring 100% compliance with global surveillance privacy laws.

🗂️ Project Structure
The system is split into a React-based command center frontend and a Python/FastAPI computer vision backend.

frontend/: React 18 application using Vite, TailwindCSS, and Recharts for live telemetry visualization.

backend/: Python edge-node engine leveraging FastAPI, Ultralytics YOLOv8, and OpenCV, streaming processed JSON data and video via WebSockets.

🛠️ Technologies & AMD Hardware Optimization
Frontend: React, TailwindCSS, Recharts, Lucide React.

Backend: Python, FastAPI, WebSockets.

Computer Vision: OpenCV, Ultralytics YOLOv8.

Hardware Acceleration: The vision pipeline and PyTorch inference are designed to be deployed on local edge servers using AMD EPYC™ processors and optimized with AMD ROCm™ (Radeon Open Compute) to ensure zero-latency, maximum FPS processing without cloud reliance.

⚙️ Prerequisites
Before running the system locally for evaluation, ensure you have:

Node.js (v18 or higher)

Python (3.9 or higher)

Git

💻 Local Setup & Execution (For Judges)
1. Start the AI Vision Backend
Navigate to the backend directory, install the computer vision dependencies, and launch the WebSocket server.

Bash
cd backend
pip install -r requirements.txt
# Start the FastAPI WebSocket server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
(Note: Ensure your environment is configured for AMD ROCm if running on supported Radeon hardware for optimal YOLO inference speeds).

2. Start the React Command Center
Open a new terminal, navigate to the frontend directory, and launch the Vite development server.

Bash
cd frontend
npm install
npm run dev
🌐 Deployment Guide
Frontend Hosting (Vercel)
The easiest way to host the React command center is using Vercel.

Push your code to GitHub.

Import your repository on Vercel.

Configure your Build Command: npm run build

Set the Output Directory: dist

Configure Environment Variables (e.g., your WebSocket backend URL):
VITE_WS_URL = wss://your-backend-url.onrender.com/ws

Deploy.

Backend Hosting (Render)
For the Python FastAPI and WebSocket backend, we recommend Render for robust containerized deployment.

Create a New Web Service on Render.

Connect your GitHub Repository.

Configure Build & Start Commands:

Root Directory: backend

Build Command: pip install -r requirements.txt

Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Deploy the service.
