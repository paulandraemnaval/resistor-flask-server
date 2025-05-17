from flask import Flask, request, send_file, jsonify, make_response
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLO model
model_path = './assets/model/my_model.onnx'
model = YOLO(model_path, task='detect')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        print(f"Received file: {file.filename}")

        # Save image temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        # Open and convert image
        img = Image.open(temp_path).convert('RGB')
        print(f"Image opened. Size: {img.size}, Mode: {img.mode}")

        # Save a copy for debug inspection (to check exactly what was received)
        debug_save_path = 'debug_saved.jpg'
        img.save(debug_save_path)
        print(f"Saved debug image to: {debug_save_path}")

        original_width, original_height = img.size

        # Run inference with a lower confidence threshold temporarily
        conf_threshold = 0.25  # lower for debugging, adjust as needed
        results = model.predict(source=temp_path, conf=conf_threshold)
        print(f"Model returned {len(results)} results.")

        processed_image = img.copy()
        draw = ImageDraw.Draw(processed_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        detections = []
        merch_counts = {model.names[i]: 0 for i in model.names}
        colors = ["red", "green", "blue", "orange", "purple"]

        total_detections = 0
        for r in results:
            boxes = r.boxes
            print(f"Number of boxes detected: {len(boxes)}")
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = model.names[cls_id]
                merch_counts[label] += 1
                total_detections += 1

                print(f"Detection: {label} conf: {conf:.2f} bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

                color = colors[cls_id % len(colors)]
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                text = f"{label} ({conf:.2f})"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                draw.rectangle(
                    [(x1, y1 - text_height - 4), (x1 + text_width, y1)],
                    fill=color
                )
                draw.text((x1, y1 - text_height - 2), text, fill="white", font=font)

                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': conf,
                    'bounding_box': [float(x1), float(y1), float(x2), float(y2)]
                })

        print(f"Total detections: {total_detections}")

        # Draw merchandise counts
        y_pos = 20
        for class_name, count in merch_counts.items():
            if count > 0:
                count_text = f"{class_name}: {count}"
                text_bbox = draw.textbbox((0, 0), count_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle(
                    [(original_width - text_width - 10, y_pos - 2), 
                     (original_width - 5, y_pos + text_height + 2)],
                    fill="black"
                )
                draw.text((original_width - text_width - 8, y_pos), count_text,
                          fill="white", font=font)
                y_pos += text_height + 10

        # Convert image to buffer
        img_io = io.BytesIO()
        processed_image.save(img_io, format='JPEG')
        img_io.seek(0)

        # Cleanup
        os.remove(temp_path)

        # Create response with image and metadata in headers
        response = make_response(send_file(img_io, mimetype='image/jpeg'))
        response.headers['X-Detections'] = json.dumps(detections)
        response.headers['X-Merch-Counts'] = json.dumps(merch_counts)
        return response

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
