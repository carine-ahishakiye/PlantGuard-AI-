# 🌿 PlantGuard AI  Crop Disease Detection

> **A farmer in Rwanda shouldn't need a laboratory to know what's destroying their maize.**
> Upload a leaf photo. Get an instant AI diagnosis. Act before it's too late.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=flat-square&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square&logo=fastapi)
![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📸 Demo

| Landing Page | Profile Panel | Disease Result |
|---|---|---|
| ![Landing](images/hero-plant.jpg) | Panel slides in from left | Instant diagnosis with advice |

**Live demo:** [Coming soon on Hugging Face Spaces](#)

---

## 🎯 What It Does

PlantGuard AI detects **37 crop diseases** across 14 plant species from a single leaf photograph. Built for smallholder farmers in Sub-Saharan Africa who lack access to agronomists, this tool puts expert-level plant diagnosis in the palm of your hand — in both English and Kinyarwanda.

**Key capabilities:**
- Upload any leaf image and receive an instant AI diagnosis
- 94% validation accuracy across 38 disease categories
- Numbered treatment advice steps per disease, in English and Kinyarwanda
- Confidence score with colour-coded severity indicator
- Optional account creation to save and track diagnosis history
- Full dark mode and Kinyarwanda language support
- Share results via WhatsApp, Twitter, or email

---

## 🧠 The Model

| Property | Value |
|---|---|
| Architecture | MobileNetV2 + Transfer Learning |
| Pre-trained on | ImageNet (1.4M images) |
| Fine-tuned on | PlantVillage Dataset |
| Training images | 54,305 leaf images |
| Disease classes | 38 categories |
| Validation accuracy | **94%** |
| Training epochs | 5 |
| Inference time | < 1 second on CPU |

MobileNetV2 was chosen deliberately — it is optimised for speed and low memory, meaning it can eventually run on mobile devices in areas with limited internet connectivity.

---

## 🌱 Detectable Diseases

<details>
<summary>Click to see all 37 detectable conditions</summary>

| Plant | Diseases Detected |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |

</details>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (index.html)            │
│   Landing Page + GitHub-style Profile Panel  │
└──────────────────────┬──────────────────────┘
                       │ HTTP / REST
┌──────────────────────▼──────────────────────┐
│           FastAPI Backend (app.py)           │
│  /predict  /auth/login  /auth/signup         │
│  /auth/me  /history    /diseases             │
└──────────┬──────────────────┬───────────────┘
           │                  │
┌──────────▼──────┐  ┌────────▼────────────┐
│  TensorFlow     │  │  SQLite Database    │
│  MobileNetV2    │  │  users + predictions│
│  plant_disease  │  │  JWT Authentication │
│  _model.h5      │  │  bcrypt passwords   │
└─────────────────┘  └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- 2GB RAM minimum (for TensorFlow)

### 1. Clone the repository

```bash
git clone https://github.com/carine-ahishakiye/PlantGuard-AI.git
cd PlantGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset (first time only)

```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

### 5. Train the model (first time only)

```bash
python train.py
```

Training takes approximately 15-30 minutes. The model will be saved to `model/plant_disease_model.h5`.

### 6. Run the server

```bash
uvicorn app:app --reload
```

### 7. Open the app

```
http://127.0.0.1:8000
```

---

## 📁 Project Structure

```
PlantGuard-AI/
│
├── app.py                      # FastAPI backend — all routes and auth
├── train.py                    # Model training script
├── backup.py                   # Database backup utility
├── index.html                  # Full frontend (single file)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # For Hugging Face Spaces deployment
│
├── model/
│   ├── plant_disease_model.h5  # Trained TensorFlow model
│   └── training_results.png    # Accuracy / loss curves
│
├── data/
│   └── plantvillage dataset/
│       └── color/              # 54,305 training images (38 folders)
│
├── images/
│   ├── hero-plant.jpg          # Landing page hero image
│   └── about-tech.jpeg         # About section image
│
├── backups/                    # Auto-generated database backups
└── plantguard.db               # SQLite database (auto-created on first run)
```

---

## 🔌 API Reference

All endpoints are documented interactively at `http://127.0.0.1:8000/docs`

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Serve the frontend |
| `POST` | `/auth/signup` | No | Create a new account |
| `POST` | `/auth/login` | No | Login, returns JWT token |
| `GET` | `/auth/me` | JWT | Get current user info |
| `POST` | `/predict` | Optional | Analyse a leaf image |
| `GET` | `/history` | JWT | Get user's prediction history |
| `GET` | `/diseases` | No | List all detectable diseases |

### Example: Predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf.jpg"
```

```json
{
  "plant": "tomato",
  "condition": "early blight",
  "confidence": 94.7,
  "raw_class": "Tomato___Early_blight",
  "prediction_time_ms": 312
}
```

---

## 🔒 Security & Privacy

- Passwords hashed with **bcrypt** (never stored in plain text)
- Authentication via **JWT tokens** (expire after 24 hours)
- Leaf images are processed **in memory only** — never written to disk
- No data is shared with third parties
- Users can delete their account and all associated data at any time

---

## 🌍 Language Support

PlantGuard AI supports **English** and **Kinyarwanda**. All disease names, advice steps, UI labels, and error messages are fully translated. Language can be toggled instantly from the profile panel without reloading.

---

## 📊 Training Results

The model was trained for 5 epochs on an 80/20 train/validation split:

| Epoch | Training Accuracy | Validation Accuracy |
|---|---|---|
| 1 | 72.3% | 81.4% |
| 2 | 87.6% | 89.2% |
| 3 | 91.8% | 92.5% |
| 4 | 93.4% | 93.8% |
| 5 | 94.9% | **94.1%** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| AI Model | TensorFlow 2.21, Keras, MobileNetV2 |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Database | SQLite, JWT (python-jose), bcrypt (passlib) |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Fonts | Playfair Display, DM Sans, Syne |
| Dataset | PlantVillage (Kaggle) |
| Deployment | Hugging Face Spaces (Docker) |

---

## 🔭 Roadmap

- [x] MobileNetV2 model — 94% accuracy
- [x] FastAPI REST backend with JWT auth
- [x] Beautiful responsive frontend
- [x] Kinyarwanda language support
- [x] Dark mode
- [x] Prediction history per user
- [x] Database backup system
- [ ] Deploy to Hugging Face Spaces
- [ ] Mobile-optimised PWA version
- [ ] Offline inference (TensorFlow Lite)
- [ ] SMS diagnosis via Africa's Talking API
- [ ] Expand to 100+ diseases

---

## 👩🏾‍💻 Author

**Carine Ahishakiye Yibukabayo**
BSc (Hons) Software Engineering — Year 3
African Leadership University, Kigali, Rwanda

> This project is part of my long-term portfolio for applying to the Erasmus Mundus Joint Master's programmes in Data Science and Artificial Intelligence (EDISS, EMAI, BDMA). It combines my passion for machine learning with a genuine need in Rwandan and African agriculture.s
---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 💚 in Kigali, Rwanda
  <br/>
  <em>"Technology should serve those who need it most."</em>
</p>