from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_image(image_path: Path) -> tuple[str, float]:
    result = model(str(image_path))[0]
    top_index = int(result.probs.top1)
    label = result.names[top_index]
    confidence = float(result.probs.top1conf)
    return label, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    image_url = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an image to classify."
        elif not allowed_file(uploaded_file.filename):
            error = "Unsupported file type. Use JPG, PNG, BMP, TIFF, or WEBP."
        else:
            safe_name = secure_filename(uploaded_file.filename)
            stored_name = f"{uuid4().hex}_{safe_name}"
            image_path = UPLOAD_DIR / stored_name
            uploaded_file.save(image_path)

            try:
                label, confidence = classify_image(image_path)
                prediction = {"label": label, "confidence": confidence}
                image_url = f"/static/uploads/{stored_name}"
            except Exception:
                image_path.unlink(missing_ok=True)
                error = "The image could not be classified. Please try another image."

    return render_template(
        "index.html",
        error=error,
        prediction=prediction,
        image_url=image_url,
    )


@app.errorhandler(413)
def request_entity_too_large(_error):
    return render_template(
        "index.html",
        error="That image is too large. Please upload a file smaller than 10 MB.",
        prediction=None,
        image_url=None,
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
