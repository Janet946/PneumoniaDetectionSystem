import os
import io
import csv
import sqlite3
from datetime import datetime
from typing import Optional
from itsdangerous import URLSafeSerializer

import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 224
MODEL_PATH = "models/pneumonia_cnn_best.h5"
VALIDATOR_PATH = "models/xray_validator.h5"
VALIDATOR_SIZE = 128

UPLOAD_DIR = "uploads"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "pneumoscan.db")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_MB = 10

# Demo admin credentials (change before submission)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Session secret (change to a long random string)
SESSION_SECRET = "CHANGE_ME_TO_A_LONG_RANDOM_SECRET"
serializer = URLSafeSerializer(SESSION_SECRET)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# APP INIT
# ============================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model once
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Load X-ray validator (optional but recommended)
validator = None
if os.path.exists(VALIDATOR_PATH):
    try:
        validator = tf.keras.models.load_model(VALIDATOR_PATH)
        print("✓ X-ray validator loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load X-ray validator: {e}")
else:
    print(f"⚠ X-ray validator not found at {VALIDATOR_PATH}. Using fallback validation only.")

# ============================================================
# DB HELPERS
# ============================================================
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    conn = db_connect()
    cur = conn.cursor()

    # Scans table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            p_pneumonia REAL NOT NULL,
            confidence_percent REAL NOT NULL
        )
    """)

    # NEW: Admins table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


db_init()

# ============================================================
# AUTH HELPERS
# ============================================================
def set_session_cookie(resp: RedirectResponse, role: str):
    token = serializer.dumps({"role": role, "ts": datetime.now().isoformat()})
    resp.set_cookie("session", token, httponly=True, samesite="lax")

def clear_session_cookie(resp: RedirectResponse):
    resp.delete_cookie("session")

def is_logged_in(request: Request) -> bool:
    token = request.cookies.get("session")
    if not token:
        return False
    try:
        data = serializer.loads(token)
        return data.get("role") == "admin"
    except BadSignature:
        return False

def require_login(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    return None

# ============================================================
# ML HELPERS
# ============================================================
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXT

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def validate_xray(image: Image.Image):
    """
    Validate whether image is a chest X-ray.
    Returns: (is_valid, confidence_score)
    """
    if validator is None:
        return basic_xray_check(image), 0.0
    
    try:
        img_resized = image.convert("RGB")
        img_resized = img_resized.resize((VALIDATOR_SIZE, VALIDATOR_SIZE))
        arr = np.array(img_resized).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        validation_prob = float(validator.predict(arr, verbose=0)[0][0])
        is_valid = validation_prob > 0.5
        return is_valid, validation_prob
    except Exception as e:
        print(f"Validation error: {e}")
        return basic_xray_check(image), 0.0

def basic_xray_check(image: Image.Image) -> bool:
    """Fallback: check image dimensions and color saturation."""
    try:
        img_rgb = image.convert("RGB")
        width, height = img_rgb.size
        if width < 100 or height < 100:
            return False
        if width > height * 2.5 or height > width * 2.5:
            return False
        arr = np.array(img_rgb)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        saturation = np.std([r, g, b])
        return saturation < 50
    except:
        return True

def predict_image(image: Image.Image):
    """Predict pneumonia and validate X-ray."""
    is_valid, validation_conf = validate_xray(image)
    x = preprocess_image(image)
    p = float(model.predict(x, verbose=0)[0][0])
    label = "PNEUMONIA" if p >= 0.5 else "NORMAL"
    confidence = (p if label == "PNEUMONIA" else (1 - p)) * 100
    return label, p, confidence, is_valid, validation_conf

def save_scan_to_db(filename: str, label: str, p_pneumonia: float, confidence_percent: float) -> int:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO scans (timestamp, filename, predicted_label, p_pneumonia, confidence_percent)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        label,
        float(p_pneumonia),
        float(confidence_percent),
    ))
    conn.commit()
    scan_id = cur.lastrowid
    conn.close()
    return scan_id

# ============================================================
# 1) LOGIN SCREEN
# ============================================================
@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        resp = RedirectResponse(url="/dashboard", status_code=302)
        set_session_cookie(resp, "admin")
        return resp

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password."},
        status_code=401
    )

@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    clear_session_cookie(resp)
    return resp

# ============================================================
# 2) DASHBOARD SCREEN
# ============================================================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    redirect = require_login(request)
    if redirect:
        return redirect

    conn = db_connect()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM scans")
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) AS pneu FROM scans WHERE predicted_label='PNEUMONIA'")
    pneu = cur.fetchone()["pneu"]

    rate = round((pneu / total) * 100, 2) if total else 0.0

    cur.execute("""
        SELECT id, timestamp, filename, predicted_label, p_pneumonia, confidence_percent
        FROM scans
        ORDER BY id DESC
        LIMIT 10
    """)
    recent = [dict(r) for r in cur.fetchall()]
    conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "total_scans": total,
            "pneumonia_scans": pneu,
            "pneumonia_rate": rate,
            "recent": recent,
        }
    )

# ============================================================
# 3) UPLOAD / PREDICTION SCREEN
# ============================================================
@app.get("/scan", response_class=HTMLResponse)
def scan_page(request: Request):
    redirect = require_login(request)
    if redirect:
        return redirect
    return templates.TemplateResponse("scan.html", {"request": request})

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    redirect = require_login(request)
    if redirect:
        return redirect

    # Validate file name/ext
    if not file.filename or not allowed_file(file.filename):
        return templates.TemplateResponse(
            "scan.html",
            {"request": request, "error": "Invalid file type. Upload JPG/PNG/WEBP."},
            status_code=400
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return templates.TemplateResponse(
            "scan.html",
            {"request": request, "error": f"File too large ({size_mb:.2f}MB). Max {MAX_FILE_MB}MB."},
            status_code=400
        )

    # Open image
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        return templates.TemplateResponse(
            "scan.html",
            {"request": request, "error": "Could not read the image. Please try another file."},
            status_code=400
        )

    # Predict (includes X-ray validation)
    label, p_pneumonia, confidence_percent, is_valid_xray, validation_conf = predict_image(image)
    
    # Check if image is valid chest X-ray
    if not is_valid_xray:
        return templates.TemplateResponse(
            "scan.html",
            {
                "request": request,
                "error": (
                    "This image does not appear to be a chest X-ray. "
                    "Please upload a valid chest X-ray image for pneumonia detection."
                ),
                "validation_failed": True
            },
            status_code=400
        )

    # Save upload
    safe_name = file.filename.replace(" ", "_")
    stamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, stamped_name)
    with open(save_path, "wb") as f:
        f.write(contents)

    # Save to DB
    scan_id = save_scan_to_db(
        filename=stamped_name,
        label=label,
        p_pneumonia=p_pneumonia,
        confidence_percent=confidence_percent
    )

    # Redirect to Result screen
    return RedirectResponse(url=f"/result/{scan_id}", status_code=302)

# ============================================================
# 4) RESULT SCREEN
# ============================================================
@app.get("/result/{scan_id}", response_class=HTMLResponse)
def result_page(request: Request, scan_id: int):
    redirect = require_login(request)
    if redirect:
        return redirect

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, filename, predicted_label, p_pneumonia, confidence_percent
        FROM scans WHERE id=?
    """, (scan_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "not_found": True},
            status_code=404
        )

    data = dict(row)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "scan": data,
            "p_percent": round(data["p_pneumonia"] * 100, 2),
            "conf": round(data["confidence_percent"], 2),
        }
    )

# ============================================================
# 5) REPORTS / HISTORY SCREEN + EXPORT CSV
# ============================================================
@app.get("/reports", response_class=HTMLResponse)
def reports_page(
    request: Request,
    label: Optional[str] = None,     # NORMAL or PNEUMONIA
    q: Optional[str] = None,         # filename search
):
    redirect = require_login(request)
    if redirect:
        return redirect

    label = (label or "").strip().upper()
    q = (q or "").strip()

    where = []
    params = []

    if label in {"NORMAL", "PNEUMONIA"}:
        where.append("predicted_label=?")
        params.append(label)

    if q:
        where.append("filename LIKE ?")
        params.append(f"%{q}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, timestamp, filename, predicted_label, p_pneumonia, confidence_percent
        FROM scans
        {where_sql}
        ORDER BY id DESC
        LIMIT 200
    """, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "rows": rows,
            "filter_label": label,
            "filter_q": q
        }
    )

@app.get("/reports/export")
def export_csv(
    request: Request,
    label: Optional[str] = None,
    q: Optional[str] = None,
):
    redirect = require_login(request)
    if redirect:
        return redirect

    label = (label or "").strip().upper()
    q = (q or "").strip()

    where = []
    params = []
    if label in {"NORMAL", "PNEUMONIA"}:
        where.append("predicted_label=?")
        params.append(label)
    if q:
        where.append("filename LIKE ?")
        params.append(f"%{q}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = db_connect()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, timestamp, filename, predicted_label, p_pneumonia, confidence_percent
        FROM scans
        {where_sql}
        ORDER BY id DESC
    """, params)
    data = cur.fetchall()
    conn.close()

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "timestamp", "filename", "predicted_label", "p_pneumonia", "confidence_percent"])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for r in data:
            writer.writerow([r["id"], r["timestamp"], r["filename"], r["predicted_label"], r["p_pneumonia"], r["confidence_percent"]])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f"pneumoscan_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
