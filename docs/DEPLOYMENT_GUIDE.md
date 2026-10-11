# Cross-Platform Deployment & Multi-System Execution Guide

**Project Title:** AI-Powered Aquifer Groundwater Level Forecasting, Risk Assessment and Smart Water Management System  
**Audience:** Evaluators, Collaborators, and Deployment Engineers  

This project is packaged with **four distinct deployment options** so you can run it on another computer (Windows, Mac, Linux), share it over a local Wi-Fi network, package it into a Docker container, or deploy it to a free public cloud server.

---

## Method 1: One-Click Runner (Easiest for Other PCs)

No complex manual terminal commands required. Simply copy or clone this project folder to the other system.

### On Another Windows PC:
1. Ensure Python 3.10+ is installed on the computer (with "Add Python to PATH" checked).
2. Double-click the file:
   ```cmd
   run_aquifer.bat
   ```
3. **What happens automatically:**
   - Detects Python.
   - Creates an isolated `.venv` virtual environment.
   - Installs all dependencies (`fastapi`, `torch`, `scikit-learn`, `xgboost`, `shap`, etc.).
   - Pre-generates the hydrogeological datasets if not already present.
   - Boots up the backend and automatically opens `http://127.0.0.1:8000/` in the default browser.

### On Another Mac or Linux PC:
1. Open Terminal in the project root folder.
2. Grant execution permission (first time only) and run:
   ```bash
   chmod +x run_aquifer.sh
   ./run_aquifer.sh
   ```
3. The script configures the virtual environment, installs dependencies, and boots the application on `http://127.0.0.1:8000/`.

---

## Method 2: Access from Other Laptops, Tablets, or Mobile Phones on the Same Wi-Fi (LAN)

Both `run_aquifer.bat` and `run_aquifer.sh` bind the FastAPI server to `0.0.0.0:8000`. This means **any device connected to the same Wi-Fi network** can open the dashboard without installing anything!

### Steps:
1. Start the project on the host laptop using `run_aquifer.bat` or `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. Find the host laptop's local IP address:
   - **Windows:** Open Command Prompt and type `ipconfig`. Look for `IPv4 Address` under your active Wi-Fi adapter (e.g. `192.168.1.15`).
   - **Mac / Linux:** Type `ifconfig` or `hostname -I`.
3. On any other laptop, tablet, or phone connected to the same Wi-Fi, open the browser and type:
   ```text
   http://<HOST_IP_ADDRESS>:8000/
   Example: http://192.168.1.15:8000/
   ```
4. **Troubleshooting Windows Firewall:** If the other device cannot connect, allow port 8000 in Windows Defender Firewall:
   - Run in Administrator PowerShell:
     ```powershell
     New-NetFirewallRule -DisplayName "Aquifer System 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
     ```

---

## Method 3: Containerized Deployment via Docker & Docker Compose

Docker ensures identical execution across any operating system or cloud virtual machine with zero dependency conflicts.

### Prerequisites:
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux).

### Run with Docker Compose (Recommended):
In the project root directory, run:
```bash
docker compose up --build
```
- Open [http://localhost:8000/](http://localhost:8000/) in your browser.
- Data generated is safely persisted in the Docker volume `aquifer-data`.
- To stop the container:
  ```bash
  docker compose down
  ```

### Run with Standard Docker CLI:
```bash
# Build the Docker image
docker build -t aquifer-ai-system .

# Run the container
docker run -d -p 8000:8000 --name aquifer-app aquifer-ai-system
```

---

## Method 4: 100% Free Public Cloud Hosting (Render.com)

Deploy this project to the cloud so anyone anywhere in the world can access it via a public HTTPS link (e.g. `https://aquifer-ai-system.onrender.com`).

### Using Render.com:
1. Push this project to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for AI Aquifer Management System"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/aquifier.git
   git push -u origin main
   ```
2. Go to [https://render.com/](https://render.com/) and create a free account.
3. Click **New +** -> **Web Service**.
4. Select your `aquifier` GitHub repository.
5. Render will automatically detect [`render.yaml`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/render.yaml) or you can set:
   - **Environment:** `Python 3`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/requirements.txt && cd backend && python -m app.core.data_generator
     ```
   - **Start Command:**
     ```bash
     cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** Free
6. Click **Deploy Web Service**.
7. In ~3 minutes, Render will provide a live public HTTPS URL accessible from any device.

---

## Summary Checklist for Multi-System Demonstration

| Target System | Recommended Method | Command / Action |
| :--- | :--- | :--- |
| **Windows PC** | Local Batch Runner | Double-click [`run_aquifer.bat`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/run_aquifer.bat) |
| **Mac / Linux** | Shell Script | Run [`./run_aquifer.sh`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/run_aquifer.sh) |
| **Same Wi-Fi (LAN)** | Network IP Binding | Open `http://<HOST_IP>:8000/` on phone/laptop |
| **Any Server / VM** | Docker Compose | Run `docker compose up --build` |
| **Public Internet** | Free Cloud Hosting | Connect GitHub repo to Render.com / Railway |
