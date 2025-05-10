from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = YOLO("./assets/model/my_model.pt")

CLASS_NAMES = [
    "resistor", "1K Ohm", "1M Ohm", "2K Ohm", "2 Ohm", "3.9K Ohm", "4.7K Ohm", "5.1K Ohm", "5.6K Ohm",
    "6.8K Ohm", "7.5K Ohm", "8.2K Ohm", "10.1K Ohm", "10 Ohm", "11M Ohm", "15 Ohm", "20K Ohm",
    "22 Ohm", "24K Ohm", "27 Ohm", "33K Ohm", "56K Ohm", "68K Ohm", "100 Ohm", "150 Ohm",
    "180K Ohm", "220K Ohm", "220 Ohm", "270K Ohm", "330 Ohm", "470 Ohm", "620 Ohm", "820 Ohm",
    "4700 MOhm"
]

# ===== TWEAKABLE PARAMETERS =====
# Bounding box settings
BOX_THICKNESS = 25               # Thickness of the bounding box lines
BOX_COLOR = (0, 255, 0)         # Box color in BGR format (Green)

# Text/label settings
FONT_SCALE = 4                # Size of the font (larger value = bigger text)
FONT_THICKNESS = 2              # Thickness of the font strokes
FONT_COLOR = (0, 0, 0)        # Text color in BGR format (Green)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX  # Font type

# Text background settings
TEXT_BG_COLOR = (0, 255, 0)       # Background color for text (Black)
TEXT_PADDING = 10               # Padding around text in pixels
USE_TEXT_BACKGROUND = True      # Whether to use a background for the text

# Confidence display settings
SHOW_CONFIDENCE = False          # Whether to display confidence score
CONFIDENCE_DECIMALS = 2         # Number of decimal places for confidence

# Scale factors - set to 0 for fixed sizes or >0 to auto-scale with image
BOX_THICKNESS_SCALE = 0         # Scale factor for box thickness (0 = use fixed BOX_THICKNESS)
FONT_SCALE_FACTOR = 0           # Scale factor for font size (0 = use fixed FONT_SCALE)
# ================================

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_np = np.array(image)
    
    # Get image dimensions
    height, width = image_np.shape[:2]
    
    # Calculate scaled parameters if scale factors are provided
    if BOX_THICKNESS_SCALE > 0:
        box_thickness = max(1, int(min(width, height) * BOX_THICKNESS_SCALE))
    else:
        box_thickness = BOX_THICKNESS
        
    if FONT_SCALE_FACTOR > 0:
        font_scale = max(0.5, min(width, height) * FONT_SCALE_FACTOR)
    else:
        font_scale = FONT_SCALE
    
    results = model(image_np)

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            conf = float(box.conf)
            cls = int(box.cls)
            class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else "Unknown"

            # Draw rectangle
            cv2.rectangle(image_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), BOX_COLOR, box_thickness)
            
            # Prepare the label text
            if SHOW_CONFIDENCE:
                label_text = f"{class_name} {conf:.{CONFIDENCE_DECIMALS}f}"
            else:
                label_text = class_name
            
            # Get the size of the text
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, FONT_FACE, font_scale, FONT_THICKNESS
            )
            
            # Draw a filled rectangle as background for the text if enabled
            if USE_TEXT_BACKGROUND:
                cv2.rectangle(
                    image_np, 
                    (xyxy[0], xyxy[1] - text_height - TEXT_PADDING),
                    (xyxy[0] + text_width + TEXT_PADDING, xyxy[1]),
                    TEXT_BG_COLOR,
                    -1  # Filled rectangle
                )
            
            # Draw the text
            cv2.putText(
                image_np, 
                label_text,
                (xyxy[0] + TEXT_PADDING//2, xyxy[1] - TEXT_PADDING//2),  # Position
                FONT_FACE,      # Font
                font_scale,     # Font scale
                FONT_COLOR,     # Text color
                FONT_THICKNESS, # Text thickness
                cv2.LINE_AA     # Line type - antialiased
            )

    # Convert back to PIL and return image
    result_image = Image.fromarray(image_np)
    buf = io.BytesIO()
    result_image.save(buf, format='JPEG')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)