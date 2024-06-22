from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import pandas as pd
import torch
import cv2
from ultralytics import YOLO
import logging
from super_gradients.training import models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the upload folder and static/annotated folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/annotated', exist_ok=True)

# Load data
foundation_df = pd.read_csv('data/foundation.csv')
lipstick_df = pd.read_csv('data/lipstick.csv')

# Convert price columns to float
foundation_df['price'] = foundation_df['price'].str.extract(r'(\d+\.\d+)').astype(float)
lipstick_df['price'] = lipstick_df['price'].str.extract(r'(\d+\.\d+)').astype(float)

# Get price ranges
foundation_price_min = foundation_df['price'].min()
foundation_price_max = foundation_df['price'].max()
lipstick_price_min = lipstick_df['price'].min()
lipstick_price_max = lipstick_df['price'].max()

# Load YOLOv8 model for lip shape detection
lip_shape_model = YOLO('models/lips/best.pt')  # YOLOv8 for lip shape segmentation

# Load YOLO NAS model for skin type detection
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
skin_type_model = models.get('yolo_nas_m', num_classes=6, checkpoint_path="models/skin/ckpt_best.pth").to(device)

# Define mapping dictionaries
skin_type_mapping = {
    'Dry-Skin': 'Dry',
    'Normal-Skin': 'Normal',
    'Oily-Skin': 'Oily'
}

skin_tone_mapping = {
    'Dark-Tone': 'Dark',
    'Medium-Tone': 'Medium',
    'Light-Tone': 'Light'
}

lip_shape_mapping = {
    'Thin-Lips': 'Thin',
    'Normal-Lips': 'Normal',
    'Full-Lips': 'Full'
}

# Function to log data to a JSON file
def log_to_json(data, filename="debug_log.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Function to analyze the image for skin type, skin tone, and lip shape
def analyze_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(path)
    annotated_path = os.path.join('static/annotated', filename)

    # Logging the paths
    app.logger.info(f"Original image path: {path}")
    app.logger.info(f"Annotated image path: {annotated_path}")

    # Skin type model prediction with explicit confidence threshold
    skin_results = skin_type_model.predict(image, conf=0.30)
    skin_predictions = skin_results.prediction

    # Debugging: Print all confidence scores
    app.logger.info(f"Confidence scores: {skin_predictions.confidence}")

    # Filter skin predictions based on confidence threshold
    valid_skin_indices = [i for i, conf in enumerate(skin_predictions.confidence) if conf >= 0.30]

    skin_bboxes = [skin_predictions.bboxes_xyxy[i].astype(int) for i in valid_skin_indices]
    skin_confidences = [skin_predictions.confidence[i].item() for i in valid_skin_indices]
    skin_class_ids = [skin_predictions.labels[i].item() for i in valid_skin_indices]
    skin_class_names = [skin_results.class_names[class_id] for class_id in skin_class_ids]

    skin_predictions_list = list(zip(skin_bboxes, skin_confidences, skin_class_names))

    # Annotate the image with skin predictions
    for bbox, confidence, class_name in skin_predictions_list:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Lip shape model prediction
    lip_results = lip_shape_model(path)
    lip_predictions = lip_results[0]

    if len(lip_predictions.masks.xy) == 0:
        lip_shape_prediction = None
    else:
        # Get the segment with the highest confidence
        highest_confidence_index = lip_predictions.boxes.conf.argmax()
        bbox = lip_predictions.boxes.xyxy[highest_confidence_index].cpu().numpy().astype(int)
        confidence = lip_predictions.boxes.conf[highest_confidence_index].item()
        class_id = lip_predictions.boxes.cls[highest_confidence_index].item()
        class_name = lip_shape_model.names[class_id]
        lip_shape_prediction = (bbox, confidence, class_name)

        # Annotate the image with lip shape prediction
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the annotated image
    cv2.imwrite(annotated_path, image)

    return skin_predictions_list, lip_shape_prediction

# Function to recommend foundation products
def recommend_foundation(skin_type, skin_tone, price_range=None, product_category=None):
    skin_type_mapped = skin_type_mapping.get(skin_type.strip(), skin_type.strip())  # Apply mapping
    skin_tone_mapped = skin_tone_mapping.get(skin_tone.strip(), skin_tone.strip())  # Apply mapping

    log_data = {
        "detected_skin_type": skin_type,
        "detected_skin_tone": skin_tone,
        "mapped_skin_type": skin_type_mapped,
        "mapped_skin_tone": skin_tone_mapped
    }

    recommendations = foundation_df[
        foundation_df['skin_type'].str.lower().str.contains(skin_type_mapped.lower()) &
        foundation_df['skin_tone'].str.lower().str.contains(skin_tone_mapped.lower())
    ]

    if price_range:
        min_price, max_price = price_range
        recommendations = recommendations[
            recommendations['price'].between(min_price, max_price)
        ]

    if product_category:
        recommendations = recommendations[
            recommendations['product_category'].str.lower() == product_category.lower()
        ]

    log_data["recommendations"] = recommendations.to_dict(orient='records')
    log_to_json(log_data, "foundation_recommendation_log.json")

    return recommendations.to_dict(orient='records')

def recommend_lipstick(skin_tone, lip_shape, price_range=None, product_category=None):
    skin_tone_mapped = skin_tone_mapping.get(skin_tone.strip(), skin_tone.strip())  # Apply mapping
    lip_shape_mapped = lip_shape_mapping.get(lip_shape.strip(), lip_shape.strip())  # Apply mapping

    log_data = {
        "detected_skin_tone": skin_tone,
        "detected_lip_shape": lip_shape,
        "mapped_skin_tone": skin_tone_mapped,
        "mapped_lip_shape": lip_shape_mapped
    }

    recommendations = lipstick_df[
        (lipstick_df['skin_tone'].str.lower() == skin_tone_mapped.lower()) &
        (lipstick_df['lip_shape'].str.lower() == lip_shape_mapped.lower())
    ]

    if price_range:
        min_price, max_price = price_range
        recommendations = recommendations[
            recommendations['price'].between(min_price, max_price)
        ]

    if product_category:
        recommendations = recommendations[
            recommendations['product_category'].str.lower() == product_category.lower()
        ]

    log_data["recommendations"] = recommendations.to_dict(orient='records')
    log_to_json(log_data, "lipstick_recommendation_log.json")

    return recommendations.to_dict(orient='records')


@app.route('/recommend/foundation', methods=['POST'])
def foundation_recommendation():
    data = request.get_json()
    filename = data.get('filename')
    price_min = data.get('price_min')
    price_max = data.get('price_max')
    product_category = data.get('product_category')
    skin_type = data.get('skin_type')
    skin_tone = data.get('skin_tone')

    app.logger.info(f"Received data: {data}")

    if not filename:
        app.logger.error('No filename provided')
        return jsonify({'error': 'No filename provided'}), 400

    if not skin_type or not skin_tone:
        app.logger.error('Skin type or skin tone not provided')
        return jsonify({'error': 'Skin type or skin tone not provided'}), 400

    price_range = (price_min, price_max) if price_min and price_max else None
    recommendations = recommend_foundation(skin_type, skin_tone, price_range, product_category)

    app.logger.info(f"Recommendations: {recommendations}")

    return jsonify(recommendations)

@app.route('/recommend/lipstick', methods=['POST'])
def lipstick_recommendation():
    data = request.get_json()
    filename = data.get('filename')
    price_min = data.get('price_min')
    price_max = data.get('price_max')
    product_category = data.get('product_category')
    skin_tone = data.get('skin_tone')
    lip_shape = data.get('lip_shape')

    app.logger.info(f"Received data: {data}")

    if not filename:
        app.logger.error('No filename provided')
        return jsonify({'error': 'No filename provided'}), 400

    if not skin_tone or not lip_shape:
        app.logger.error('Skin tone or lip shape not provided')
        return jsonify({'error': 'Skin tone or lip shape not provided'}), 400

    price_range = (price_min, price_max) if price_min and price_max else None
    recommendations = recommend_lipstick(skin_tone, lip_shape, price_range, product_category)

    app.logger.info(f"Recommendations: {recommendations}")

    return jsonify(recommendations)

@app.route('/price_range', methods=['GET'])
def price_range():
    return jsonify({
        'foundation': {'min': foundation_price_min, 'max': foundation_price_max},
        'lipstick': {'min': lipstick_price_min, 'max': lipstick_price_max}
    })

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info('Files: %s', request.files)
    if 'file' not in request.files:
        app.logger.error('No file part')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File successfully uploaded', 'filename': filename}), 200
    else:
        app.logger.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')
    app.logger.info('Received filename: %s', filename)  # Log the filename

    if not filename:
        app.logger.error('No filename provided')
        return jsonify({'error': 'No filename provided'}), 400

    # Analyze the image for skin type, skin tone, and lip shape
    skin_predictions, lip_shape_prediction = analyze_image(filename)

    if len(skin_predictions) == 0 and not lip_shape_prediction:
        app.logger.error('No detections found')
        return jsonify({'error': 'No detections found'}), 400

    # Identify and separate skin type and skin tone
    detected_skin_type = ""
    detected_skin_tone = ""
    for _, _, skin_class_name in skin_predictions:
        if 'Tone' in skin_class_name:
            detected_skin_tone = skin_class_name.replace('-Tone', '').strip()
        elif 'Skin' in skin_class_name:
            detected_skin_type = skin_class_name.replace('-Skin', '').strip()

    skin_type = detected_skin_type
    skin_tone = detected_skin_tone

    lip_shape = ""
    if lip_shape_prediction:
        _, _, lip_shape_class_name = lip_shape_prediction
        lip_shape = lip_shape_class_name

    response_data = {
        'filename': filename,
        'skin_type': skin_type,
        'skin_tone': skin_tone,
        'lip_shape': lip_shape
    }

    app.logger.info('Analysis result: %s', response_data)  # Log the analysis result

    return jsonify(response_data)

@app.route('/foundation-brands', methods=['GET'])
def get_foundation_brands():
    df = pd.read_csv('data/foundation.csv')
    brands = df['product_category'].unique().tolist()
    return jsonify(brands)

@app.route('/lipstick-brands', methods=['GET'])
def get_lipstick_brands():
    df = pd.read_csv('data/lipstick.csv')
    brands = df['product_category'].unique().tolist()
    return jsonify(brands)

@app.route('/annotated/<filename>')
def serve_annotated_image(filename):
    return send_from_directory('static/annotated', filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

