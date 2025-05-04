from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import uuid
import logging
import traceback
import subprocess
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='', static_folder='static')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

################## IMAGE DENOISING CONFIG ##################
# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
NOISE_FACTOR = 0.3
ENSEMBLE_SIZE = 3

# PSNR metric
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

# Combined loss
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.8 * mse + 0.2 * ssim

# Load Image model
IMAGE_MODEL_PATH = r"Image\denoising_model_best.keras"
try:
    image_model = tf.keras.models.load_model(
        IMAGE_MODEL_PATH,
        custom_objects={'psnr_metric': psnr_metric, 'combined_loss': combined_loss}
    )
    logger.info(f"Loaded image model from {IMAGE_MODEL_PATH}")
    image_model.summary()
except Exception as e:
    logger.error(f"Error loading image model: {e}")
    image_model = None

################## AUDIO DENOISING CONFIG ##################
# Check for ffmpeg availability
def check_ffmpeg():
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffprobe not found. Audio format conversion may fail.")
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

# Load Audio model
AUDIO_MODEL_PATH = r"Audio\audio_denoiser_final.keras"
try:
    if not os.path.exists(AUDIO_MODEL_PATH):
        logger.warning(f"Audio model file not found at: {os.path.abspath(AUDIO_MODEL_PATH)}")
        audio_model = None
    else:
        logger.info(f"Loading audio model from {AUDIO_MODEL_PATH}")
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        logger.info("Audio model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load audio model: {str(e)}")
    audio_model = None

################## IMAGE PROCESSING FUNCTIONS ##################
def preprocess_image(img_data, noise_type="gaussian", noise_factor=0.3, denoise_only=False):
    # Convert bytes to image
    img = Image.open(io.BytesIO(img_data))
    img = np.array(img)
    
    # Save original shape for resizing back
    original_shape = img.shape[:2]
    
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
        
    # Resize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    
    if denoise_only:
        # In denoise-only mode, use the input image as the noisy image
        noisy_img = img
    else:
        # Add noise
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_factor, img.shape)
            noisy_img = img + noise
        elif noise_type == "salt_pepper":
            noisy_img = img.copy()
            # Salt
            salt_mask = np.random.random(img.shape) < (noise_factor / 2)
            noisy_img[salt_mask] = 1.0
            # Pepper
            pepper_mask = np.random.random(img.shape) < (noise_factor / 2)
            noisy_img[pepper_mask] = 0.0
        elif noise_type == "poisson":
            # Scale image for Poisson noise
            scaled_img = img * 255.0 * noise_factor
            # Add Poisson noise and rescale
            noisy_img = np.random.poisson(scaled_img) / (255.0 * noise_factor)
        else:
            # Default to Gaussian
            noise = np.random.normal(0, noise_factor, img.shape)
            noisy_img = img + noise
        
        # Clip values
        noisy_img = np.clip(noisy_img, 0.0, 1.0)
    
    return img, noisy_img

def process_image(img_data, noise_type, noise_factor, denoise_only):
    # Check if model is loaded
    if image_model is None:
        return {"error": "Image denoising model not loaded"}
    
    # Preprocess
    original_img, noisy_img = preprocess_image(img_data, noise_type, float(noise_factor), denoise_only)
    
    # Predict with ensemble
    predictions = []
    for _ in range(ENSEMBLE_SIZE):
        input_tensor = tf.expand_dims(noisy_img, axis=0)
        pred = image_model.predict(input_tensor, verbose=0)
        predictions.append(pred[0])
    
    # Average ensemble predictions
    denoised_img = np.mean(predictions, axis=0)
    
    # Apply mild post-processing for better visual results
    denoised_img = cv2.GaussianBlur(denoised_img, (3, 3), 0.5)
    denoised_img = np.clip(denoised_img, 0, 1)
    
    # Calculate PSNR (ensure consistent dimensions)
    psnr_val = peak_signal_noise_ratio(original_img, denoised_img, data_range=1.0)
    
    # Convert to bytes for sending to frontend
    def img_to_base64(img):
        img_array = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG", optimize=True, quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "original": img_to_base64(original_img),
        "noisy": img_to_base64(noisy_img),
        "denoised": img_to_base64(denoised_img),
        "psnr": f"{psnr_val:.2f}"
    }

################## AUDIO PROCESSING FUNCTIONS ##################
def process_audio(audio_file, sr=16000, chunk_duration=3):
    # Check if model is loaded
    if audio_model is None:
        raise ValueError("Audio denoising model not loaded")
    
    try:
        # Validate file size
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        logger.info(f"Audio file: {audio_file}, Size: {file_size} bytes")
        if file_size < 100:
            raise ValueError("Audio file is too small or invalid (likely not a valid audio file)")
        
        # Convert to WAV if ffmpeg is available
        temp_wav = os.path.join(OUTPUT_DIR, f"converted_{uuid.uuid4()}.wav")
        if FFMPEG_AVAILABLE:
            try:
                logger.info("Converting audio to WAV")
                audio_segment = AudioSegment.from_file(audio_file)
                audio_segment.export(temp_wav, format="wav")
                audio_file = temp_wav
                logger.info(f"Converted audio to WAV: {temp_wav}")
            except Exception as e:
                logger.warning(f"Failed to convert audio with pydub: {str(e)}. Falling back to direct loading.")
        else:
            logger.warning("ffmpeg not available. Attempting to load audio directly with librosa.")
        
        # Load audio
        logger.info("Loading audio with librosa")
        audio, _ = librosa.load(audio_file, sr=sr, dtype=np.float64)
        logger.info(f"Audio loaded, length: {len(audio)} samples, duration: {len(audio)/sr:.2f} seconds")
        
        # Validate audio
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be loaded")
        
        # Process audio in 3-second chunks
        chunk_length = chunk_duration * sr  # Samples per chunk (e.g., 3 * 16000 = 48000)
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        denoised_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}, length: {len(chunk)} samples")
            
            # Pad chunk if shorter than 3 seconds
            if len(chunk) < chunk_length:
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant', constant_values=0)
                logger.info(f"Padded chunk {i+1} to {len(chunk)} samples")
            
            # Generate spectrogram
            logger.info(f"Generating spectrogram for chunk {i+1}")
            spec = librosa.stft(chunk, n_fft=512, hop_length=128)
            mag, phase = np.abs(spec), np.angle(spec)
            logger.info(f"Spectrogram shape: {mag.shape}")
            
            # Validate spectrogram shape
            if mag.shape[0] != 257:
                raise ValueError(f"Unexpected frequency bins in chunk {i+1}: {mag.shape[0]}, expected 257")
            
            # Normalize magnitude
            logger.info(f"Normalizing spectrogram for chunk {i+1}")
            mag_norm = np.log1p(mag)
            mag_norm = (mag_norm - np.min(mag_norm)) / (np.max(mag_norm) - np.min(mag_norm))
            
            # Pad or crop to target shape [376, 257]
            target_time_steps = 376
            if mag_norm.shape[1] < target_time_steps:
                mag_norm = np.pad(mag_norm, ((0, 0), (0, target_time_steps - mag_norm.shape[1])), mode='constant')
            else:
                mag_norm = mag_norm[:, :target_time_steps]
            logger.info(f"Normalized spectrogram shape: {mag_norm.shape}")
            
            # Prepare input for model
            X = mag_norm.T[np.newaxis, :, :, np.newaxis]  # Shape: [1, 376, 257, 1]
            logger.info(f"Model input shape for chunk {i+1}: {X.shape}")
            
            # Validate input shape
            if X.shape != (1, 376, 257, 1):
                raise ValueError(f"Invalid input shape for chunk {i+1}: {X.shape}, expected (1, 376, 257, 1)")
            
            # Predict denoised spectrogram
            logger.info(f"Running model prediction for chunk {i+1}")
            try:
                pred_mag_norm = audio_model.predict(X, verbose=0)[0, :, :, 0].T  # Shape: [257, 376]
                logger.info(f"Prediction shape: {pred_mag_norm.shape}")
            except Exception as e:
                logger.error(f"Model prediction failed for chunk {i+1}: {str(e)}")
                raise
            
            # Ensure pred_mag_norm matches phase shape
            if pred_mag_norm.shape[1] < phase.shape[1]:
                pred_mag_norm = np.pad(pred_mag_norm, ((0, 0), (0, phase.shape[1] - pred_mag_norm.shape[1])), mode='constant')
            elif pred_mag_norm.shape[1] > phase.shape[1]:
                pred_mag_norm = pred_mag_norm[:, :phase.shape[1]]
            logger.info(f"Adjusted prediction shape: {pred_mag_norm.shape}")
            
            # Denormalize
            logger.info(f"Denormalizing spectrogram for chunk {i+1}")
            pred_mag = np.expm1(pred_mag_norm * (np.max(np.log1p(mag)) - np.min(np.log1p(mag))) + np.min(np.log1p(mag)))
            
            # Reconstruct audio
            logger.info(f"Reconstructing audio for chunk {i+1}")
            pred_spec = pred_mag * np.exp(1j * phase)
            pred_chunk = librosa.istft(pred_spec, hop_length=128)
            logger.info(f"Denoised chunk {i+1} length: {len(pred_chunk)} samples")
            
            denoised_chunks.append(pred_chunk)
        
        # Concatenate denoised chunks
        logger.info("Concatenating denoised chunks")
        pred_audio = np.concatenate(denoised_chunks)
        logger.info(f"Final denoised audio length: {len(pred_audio)} samples, duration: {len(pred_audio)/sr:.2f} seconds")
        
        # Trim to original audio length to avoid padding artifacts
        if len(pred_audio) > len(audio):
            pred_audio = pred_audio[:len(audio)]
        logger.info(f"Trimmed denoised audio to original length: {len(pred_audio)} samples")
        
        # Save denoised audio
        output_filename = os.path.join(OUTPUT_DIR, f'denoised_{uuid.uuid4()}.wav')
        logger.info(f"Saving denoised audio to {output_filename}")
        sf.write(output_filename, pred_audio, sr)
        
        return output_filename
    
    finally:
        # Clean up temporary WAV file
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            os.remove(temp_wav)

################## ROUTES ##################
@app.route('/')
def index():
    return render_template('index.html')

# Image denoising endpoints
@app.route('/image/denoise', methods=['POST'])
def image_denoise():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    noise_type = request.form.get('noise_type', 'gaussian')
    noise_factor = request.form.get('noise_factor', '0.3')
    denoise_only = request.form.get('denoise_only', 'false').lower() == 'true'
    
    try:
        # Read image file once and pass the raw data
        img_data = image_file.read()
        result = process_image(img_data, noise_type, noise_factor, denoise_only)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/image/download', methods=['POST'])
def image_download():
    try:
        img_data = request.json.get('imageData')
        if not img_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Convert base64 to image
        img_data = img_data.split(',')[1] if ',' in img_data else img_data  # Remove the data:image/png;base64, part
        img_bytes = base64.b64decode(img_data)
        
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='denoised_image.png'
        )
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Audio denoising endpoints
@app.route('/audio/denoise', methods=['POST'])
def audio_denoise():
    if 'audio' not in request.files:
        logger.error("No audio file provided in request")
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    temp_filename = os.path.join(OUTPUT_DIR, f'temp_{uuid.uuid4()}.wav')
    audio_file.save(temp_filename)
    
    try:
        logger.info(f"Processing audio file: {temp_filename}")
        denoised_filename = process_audio(temp_filename)
        logger.info(f"Denoised audio saved to: {denoised_filename}")
        return send_file(denoised_filename, mimetype='audio/wav', as_attachment=False)
    except Exception as e:
        logger.error(f"Error during denoising: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    
    # Print статус моделей
    logger.info(f"Image model status: {'Loaded' if image_model is not None else 'Not loaded'}")
    logger.info(f"Audio model status: {'Loaded' if audio_model is not None else 'Not loaded'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)