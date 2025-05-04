import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization, Cropping2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm
import urllib.request
import tarfile
import random
import glob
import uuid

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Directories
BASE_DIR = './audio_denoising'
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLEAN_DIR = os.path.join(DATA_DIR, 'clean')
NOISY_DIR = os.path.join(DATA_DIR, 'noisy')
NOISE_DIR = os.path.join(DATA_DIR, 'noise')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Download and extract LibriSpeech dataset
def download_librispeech():
    print("Downloading LibriSpeech dataset...")
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    download_path = os.path.join(DATA_DIR, "dev-clean.tar.gz")
    
    if not os.path.exists(download_path):
        urllib.request.urlretrieve(url, download_path)
        
    print("Extracting LibriSpeech dataset...")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    
    return os.path.join(DATA_DIR, "LibriSpeech", "dev-clean")

# Generate synthetic noise samples
def download_noise():
    print("Generating synthetic noise samples...")
    for i in range(10):
        # Generate white noise
        white_noise = np.random.normal(0, 1, 16000 * 5)
        sf.write(os.path.join(NOISE_DIR, f'white_noise_{i}.wav'), white_noise, 16000)
        
        # Generate colored noise
        colored_noise = np.random.normal(0, 1, 16000 * 5)
        colored_noise = librosa.effects.preemphasis(colored_noise)
        sf.write(os.path.join(NOISE_DIR, f'colored_noise_{i}.wav'), colored_noise, 16000)
        
        # Generate impulse noise
        impulse_noise = np.zeros(16000 * 5)
        impulse_positions = np.random.randint(0, 16000 * 5, 100)
        impulse_noise[impulse_positions] = np.random.normal(0, 10, 100)
        sf.write(os.path.join(NOISE_DIR, f'impulse_noise_{i}.wav'), impulse_noise, 16000)

# Prepare data: Extract audio files, add noise
def prepare_data(librispeech_dir, max_files=200):
    all_audio_files = []
    for root, dirs, files in os.walk(librispeech_dir):
        for file in files:
            if file.endswith('.flac'):
                all_audio_files.append(os.path.join(root, file))
    
    random.shuffle(all_audio_files)
    audio_files = all_audio_files[:max_files]
    
    print(f"Processing {len(audio_files)} audio files...")
    
    noise_files = glob.glob(os.path.join(NOISE_DIR, '*.wav'))
    
    for i, file_path in enumerate(tqdm(audio_files)):
        clean_audio, sr = librosa.load(file_path, sr=16000)
        
        if len(clean_audio) < 3 * sr:
            clean_audio = np.pad(clean_audio, (0, 3 * sr - len(clean_audio)))
        
        clean_audio = clean_audio[:3 * sr]
        
        clean_path = os.path.join(CLEAN_DIR, f'clean_{i}.wav')
        sf.write(clean_path, clean_audio, sr)
        
        noise_file = random.choice(noise_files)
        noise, _ = librosa.load(noise_file, sr=sr)
        
        if len(noise) < len(clean_audio):
            repetitions = int(np.ceil(len(clean_audio) / len(noise)))
            noise = np.tile(noise, repetitions)
        
        noise = noise[:len(clean_audio)]
        
        snr = random.uniform(5, 15)
        clean_rms = np.sqrt(np.mean(clean_audio**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        noise_gain = clean_rms / (noise_rms * 10**(snr/20))
        
        noisy_audio = clean_audio + noise * noise_gain
        
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        noisy_path = os.path.join(NOISY_DIR, f'noisy_{i}.wav')
        sf.write(noisy_path, noisy_audio, sr)

# Generate spectrograms with fixed dimensions
def generate_spectrograms(max_samples=200):
    clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, '*.wav')))
    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, '*.wav')))
    
    clean_files = clean_files[:max_samples]
    noisy_files = noisy_files[:max_samples]
    
    X = []
    Y = []
    target_time_steps = 376  # Adjusted to match U-Net architecture
    target_freq_bins = 257   # n_fft=512 -> 512/2 + 1 = 257
    
    print("Generating spectrograms...")
    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        clean_audio, sr = librosa.load(clean_file, sr=16000)
        noisy_audio, sr = librosa.load(noisy_file, sr=16000)
        
        clean_spec = librosa.stft(clean_audio, n_fft=512, hop_length=128)
        noisy_spec = librosa.stft(noisy_audio, n_fft=512, hop_length=128)
        
        clean_mag = np.log1p(np.abs(clean_spec))
        noisy_mag = np.log1p(np.abs(noisy_spec))
        
        clean_mag = (clean_mag - np.min(clean_mag)) / (np.max(clean_mag) - np.min(clean_mag))
        noisy_mag = (noisy_mag - np.min(noisy_mag)) / (np.max(noisy_mag) - np.min(noisy_mag))
        
        # Pad or crop to target dimensions
        if clean_mag.shape[1] < target_time_steps:
            clean_mag = np.pad(clean_mag, ((0, 0), (0, target_time_steps - clean_mag.shape[1])), mode='constant')
            noisy_mag = np.pad(noisy_mag, ((0, 0), (0, target_time_steps - noisy_mag.shape[1])), mode='constant')
        else:
            clean_mag = clean_mag[:, :target_time_steps]
            noisy_mag = noisy_mag[:, :target_time_steps]
        
        clean_mag = clean_mag.T[:, :, np.newaxis]
        noisy_mag = noisy_mag.T[:, :, np.newaxis]
        
        X.append(noisy_mag)
        Y.append(clean_mag)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

# U-Net model with ResNet-style blocks
def build_unet_model(input_shape):
    def encoder_block(inputs, filters, kernel_size=3, batch_norm=True):
        x = Conv2D(filters, kernel_size, padding='same')(inputs)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        skip = Conv2D(filters, 1, padding='same')(inputs)
        x = tf.keras.layers.add([x, skip])
        return x, x
    
    def decoder_block(inputs, skip_features, filters, kernel_size=3, batch_norm=True):
        x = UpSampling2D((2, 2))(inputs)
        
        # Crop skip_features to match x's dimensions (time and frequency)
        if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
            crop_time = max(0, skip_features.shape[1] - x.shape[1])
            crop_freq = max(0, skip_features.shape[2] - x.shape[2])
            if crop_time > 0 or crop_freq > 0:
                skip_features = Cropping2D(cropping=((0, crop_time), (0, crop_freq)))(skip_features)
        
        x = Concatenate()([x, skip_features])
        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        skip = Conv2D(filters, 1, padding='same')(x)
        x = tf.keras.layers.add([x, skip])
        return x
    
    inputs = Input(input_shape)
    
    e1, skip1 = encoder_block(inputs, 32, batch_norm=False)
    p1 = MaxPooling2D((2, 2))(e1)
    
    e2, skip2 = encoder_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(e2)
    
    e3, skip3 = encoder_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(e3)
    
    e4, skip4 = encoder_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(e4)
    
    b, _ = encoder_block(p4, 512)
    
    d1 = decoder_block(b, skip4, 256)
    d2 = decoder_block(d1, skip3, 128)
    d3 = decoder_block(d2, skip2, 64)
    d4 = decoder_block(d3, skip1, 32)
    
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    
    # Pad output to match target shape [376, 257, 1]
    outputs = ZeroPadding2D(padding=((4, 4), (0, 1)))(outputs)  # Pad 4 top, 4 bottom, 0 left, 1 right
    
    model = Model(inputs, outputs)
    return model

# Training function
def train_model(X, Y, val_split=0.2):
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    input_shape = X_train[0].shape
    
    model = build_unet_model(input_shape)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'audio_denoiser_best.keras'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=16,
        epochs=50,
        callbacks=callbacks
    )
    
    model.save(os.path.join(MODEL_DIR, 'audio_denoiser_final.keras'))
    
    return model, history

# Evaluate model with sample audio
def evaluate_model(model, sample_idx=0):
    clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, '*.wav')))
    noisy_files = sorted(glob.glob(os.path.join(NOISY_DIR, '*.wav')))
    
    if sample_idx >= len(clean_files):
        sample_idx = 0
    
    clean_audio, sr = librosa.load(clean_files[sample_idx], sr=16000)
    noisy_audio, sr = librosa.load(noisy_files[sample_idx], sr=16000)
    
    clean_spec = librosa.stft(clean_audio, n_fft=512, hop_length=128)
    noisy_spec = librosa.stft(noisy_audio, n_fft=512, hop_length=128)
    
    clean_mag, clean_phase = np.abs(clean_spec), np.angle(clean_spec)
    noisy_mag, noisy_phase = np.abs(noisy_spec), np.angle(noisy_spec)
    
    clean_mag_norm = (np.log1p(clean_mag) - np.min(np.log1p(clean_mag))) / (np.max(np.log1p(clean_mag)) - np.min(np.log1p(clean_mag)))
    noisy_mag_norm = (np.log1p(noisy_mag) - np.min(np.log1p(noisy_mag))) / (np.max(np.log1p(noisy_mag)) - np.min(np.log1p(noisy_mag)))
    
    target_time_steps = 376
    if noisy_mag_norm.shape[1] < target_time_steps:
        noisy_mag_norm = np.pad(noisy_mag_norm, ((0, 0), (0, target_time_steps - noisy_mag_norm.shape[1])), mode='constant')
    else:
        noisy_mag_norm = noisy_mag_norm[:, :target_time_steps]
    
    X_sample = noisy_mag_norm.T[np.newaxis, :, :, np.newaxis]
    
    pred_mag_norm = model.predict(X_sample)[0, :, :, 0].T
    
    # Ensure pred_mag_norm matches noisy_phase shape
    if pred_mag_norm.shape[1] < noisy_phase.shape[1]:
        pred_mag_norm = np.pad(pred_mag_norm, ((0, 0), (0, noisy_phase.shape[1] - pred_mag_norm.shape[1])), mode='constant')
    elif pred_mag_norm.shape[1] > noisy_phase.shape[1]:
        pred_mag_norm = pred_mag_norm[:, :noisy_phase.shape[1]]
    
    pred_mag = np.expm1(pred_mag_norm * (np.max(np.log1p(noisy_mag)) - np.min(np.log1p(noisy_mag))) + np.min(np.log1p(noisy_mag)))
    
    pred_spec = pred_mag * np.exp(1j * noisy_phase)
    
    pred_audio = librosa.istft(pred_spec, hop_length=128)
    
    min_len = min(len(clean_audio), len(noisy_audio), len(pred_audio))
    clean_audio = clean_audio[:min_len]
    noisy_audio = noisy_audio[:min_len]
    pred_audio = pred_audio[:min_len]
    
    sf.write(os.path.join(BASE_DIR, 'denoised_sample.wav'), pred_audio, sr)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(noisy_mag), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(pred_mag), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
    plt.title('Predicted Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(clean_mag), sr=sr, hop_length=128, x_axis='time', y_axis='hz')
    plt.title('Ground Truth Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'spectrogram_comparison.png'))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(noisy_audio, sr=sr)
    plt.title('Noisy Audio')
    
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(pred_audio, sr=sr)
    plt.title('Denoised Audio')
    
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(clean_audio, sr=sr)
    plt.title('Clean Audio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'waveform_comparison.png'))
    
    return clean_audio, noisy_audio, pred_audio

def main():
    librispeech_dir = download_librispeech()
    download_noise()
    prepare_data(librispeech_dir, max_files=200)
    X, Y = generate_spectrograms(max_samples=200)
    model, history = train_model(X, Y)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    
    clean_audio, noisy_audio, pred_audio = evaluate_model(model)
    
    print("Training complete! Results are saved in:", BASE_DIR)
    print("- Model saved at:", os.path.join(MODEL_DIR, 'audio_denoiser_final.keras'))
    print("- Denoised audio sample saved at:", os.path.join(BASE_DIR, 'denoised_sample.wav'))
    print("- Visualization plots saved in:", BASE_DIR)

if __name__ == "__main__":
    main()