import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from datetime import datetime

from test import enhance_image

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === Flask Setup ===
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# === Load SRGAN Generator ===
from model import Generator  # Make sure you have model.py as discussed before

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

generator = Generator().to(device)
model_path = 'model.pth'

if not os.path.exists(model_path):
    logger.error(f"Model file '{model_path}' not found! Please place it in the project root.")
else:
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    logger.info("SRGAN generator model loaded successfully.")

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    start_time = datetime.now()
    logger.info("Received image upload request.")

    if 'file' not in request.files:
        logger.warning("Upload failed: No file part in request.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload failed: No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    logger.info(f"File saved: {upload_path}")

    try:
        output_filename = f"enhanced_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        logger.info("Starting image enhancement...")
        enhance_image(input_image=upload_path, output_image=output_path, generator=generator)
        logger.info(f"Image enhancement completed: {output_path}")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {duration:.2f}s")

        return jsonify({
            'message': 'Enhancement successful',
            'file_url': f'/{os.path.relpath(upload_path, start=".")}',
            'enhanced_url': f'/{os.path.relpath(output_path, start=".")}'
        })

    except Exception as e:
        logger.exception("Error during image enhancement.")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)