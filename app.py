import os
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn

# Config
UPLOAD_FOLDER = 'static/uploads'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['oblong', 'oval', 'round', 'square']

NUM_CLASSES = 4
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

# Load weights
model.load_state_dict(torch.load("face_shape_classifier2.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


# Load MTCNN
detector = MTCNN(keep_all=False, device=DEVICE)

# App setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
def predict_and_annotate(image_path):
    image = Image.open(image_path).convert("RGB")
    boxes, _ = detector.detect(image)

    if boxes is None:
        return None

    # Use first face
    x1, y1, x2, y2 = boxes[0]
    margin_ratio = 0.5
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio
    hx1, hy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
    hx2, hy2 = min(int(x2 + mx), image.width), min(int(y2 + my), image.height)

    cropped = image.crop((hx1, hy1, hx2, hy2)).resize((160, 160))
    tensor = transforms.ToTensor()(cropped).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[0][pred_idx].item()

    # Annotate image
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
    draw.text((x1, y1 - 10), f"{pred_class} ({confidence:.2f})", fill="green")

    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated.jpg')
    image.save(annotated_path)

    return annotated_path, pred_class, f"{confidence*100:.2f}%"


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result = predict_and_annotate(filepath)
            if result:
                result_img, label, confidence = result
                return render_template("index.html", result_img=result_img, label=label, confidence=confidence)
            else:
                return render_template("index.html", error="Face not detected")


    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
