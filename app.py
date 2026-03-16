from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io, sqlite3, time
from datetime import datetime, timedelta
from typing import Optional

# ── Security config ──────────────────────────────────────────
SECRET_KEY = "plantguard-secret-key-change-in-production-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# ── Database setup ────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("plantguard.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            plant TEXT,
            condition TEXT,
            confidence REAL,
            raw_class TEXT,
            prediction_time REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Pydantic models ───────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    user_email: str

class TokenData(BaseModel):
    email: Optional[str] = None

# ── Auth helpers ──────────────────────────────────────────────
def get_user(email: str):
    conn = sqlite3.connect("plantguard.db")
    c = conn.cursor()
    c.execute("SELECT id, name, email, password FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def verify_password(plain, hashed):
    return pwd_context.verify(plain[:72], hashed)  # bcrypt max 72 bytes

def hash_password(password):
    return pwd_context.hash(password[:72])  # bcrypt max 72 bytes

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        user = get_user(email)
        return user
    except JWTError:
        return None

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="PlantGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

print("Loading model...")
model = tf.keras.models.load_model("model/plant_disease_model.h5")
with open("class_names.json", "r") as f:
    class_names = json.load(f)
print(f"Model loaded! Can detect {len(class_names)} diseases.")

# ── Image prep ────────────────────────────────────────────────
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("index.html")

# Auth routes
@app.post("/auth/signup", response_model=Token)
def signup(user: UserCreate):
    existing = get_user(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(user.password)
    conn = sqlite3.connect("plantguard.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (user.name, user.email, hashed)
    )
    conn.commit()
    conn.close()
    token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_name": user.name,
        "user_email": user.email
    }

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user[3]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    token = create_access_token(
        data={"sub": user[2]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_name": user[1],
        "user_email": user[2]
    }

@app.get("/auth/me")
def get_me(current_user=Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": current_user[0], "name": current_user[1], "email": current_user[2]}

# Predict route
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    contents = await file.read()
    start_time = time.time()
    img_array = prepare_image(contents)
    predictions = model.predict(img_array)
    prediction_time = round((time.time() - start_time) * 1000)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    parts = predicted_class.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    # Save to history if user is logged in
    if current_user:
        conn = sqlite3.connect("plantguard.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO predictions (user_id, plant, condition, confidence, raw_class, prediction_time) VALUES (?, ?, ?, ?, ?, ?)",
            (current_user[0], plant, condition, round(confidence, 2), predicted_class, prediction_time)
        )
        conn.commit()
        conn.close()

    return {
        "plant": plant,
        "condition": condition,
        "confidence": round(confidence, 2),
        "raw_class": predicted_class,
        "prediction_time_ms": prediction_time
    }

# History route
@app.get("/history")
def get_history(current_user=Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = sqlite3.connect("plantguard.db")
    c = conn.cursor()
    c.execute(
        "SELECT plant, condition, confidence, raw_class, prediction_time, created_at FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
        (current_user[0],)
    )
    rows = c.fetchall()
    conn.close()
    return {
        "history": [
            {
                "plant": r[0],
                "condition": r[1],
                "confidence": r[2],
                "raw_class": r[3],
                "prediction_time_ms": r[4],
                "created_at": r[5]
            }
            for r in rows
        ]
    }

# Diseases route
@app.get("/diseases")
def list_diseases():
    return {"total": len(class_names), "diseases": class_names}

# Privacy policy
@app.get("/privacy")
def privacy():
    return FileResponse("privacy.html")