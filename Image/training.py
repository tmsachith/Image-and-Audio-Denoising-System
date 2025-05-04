import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Add, Multiply
from skimage.metrics import peak_signal_noise_ratio
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Check for GPU
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50
NOISE_FACTOR_RANGE = (0.1, 0.5)

# Load DIV2K dataset
def load_div2k_dataset():
    try:
        dataset = tfds.load('div2k/bicubic_x4', split='train', as_supervised=False)
        print("DIV2K dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"DIV2K not available: {e}. Falling back to CIFAR-10.")
        (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        return dataset

# Image preprocessing
def preprocess_image(image):
    if isinstance(image, dict) and 'hr' in image:
        image = image['hr']
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1, 1, 3])
    return image

# Add variable noise
def add_noise(image):
    noise_factor = tf.random.uniform((), NOISE_FACTOR_RANGE[0], NOISE_FACTOR_RANGE[1])
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy_image, image

# Enhanced data augmentation
def augment(noisy_image, clean_image):
    if tf.random.uniform(()) > 0.5:
        noisy_image = tf.image.flip_left_right(noisy_image)
        clean_image = tf.image.flip_left_right(clean_image)
    if tf.random.uniform(()) > 0.5:
        noisy_image = tf.image.flip_up_down(noisy_image)
        clean_image = tf.image.flip_up_down(clean_image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    noisy_image = tf.image.rot90(noisy_image, k)
    clean_image = tf.image.rot90(clean_image, k)
    noisy_image = tf.image.random_brightness(noisy_image, max_delta=0.1)
    noisy_image = tf.image.random_contrast(noisy_image, 0.8, 1.2)
    return noisy_image, clean_image

# Prepare the dataset
def prepare_dataset(dataset, augment_data=True):
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# PSNR metric
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

# Combined MSE + SSIM loss
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.8 * mse + 0.2 * ssim

# Residual block with shortcut adjustment
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same')(shortcut)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

# Attention gate
def attention_gate(input_x, gating, filters):
    x = Conv2D(filters, 1, padding='same')(input_x)
    g = Conv2D(filters, 1, padding='same')(gating)
    x = Add()([x, g])
    x = ReLU()(x)
    x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    return Multiply()([input_x, x])

# Improved U-Net model
def build_unet_denoising_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = residual_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Middle
    conv4 = residual_block(pool3, 512)
    conv4 = Dropout(0.2)(conv4)

    # Decoder with attention
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, padding='same', activation='relu')(up5)
    att5 = attention_gate(conv3, up5, 256)
    merge5 = Concatenate()([att5, up5])
    conv5 = residual_block(merge5, 256)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, padding='same', activation='relu')(up6)
    att6 = attention_gate(conv2, up6, 128)
    merge6 = Concatenate()([att6, up6])
    conv6 = residual_block(merge6, 128)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, padding='same', activation='relu')(up7)
    att7 = attention_gate(conv1, up7, 64)
    merge7 = Concatenate()([att7, up7])
    conv7 = residual_block(merge7, 64)

    outputs = Conv2D(3, 1, activation='sigmoid')(conv7)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Main execution
def main():
    start_time = time.time()

    print("Loading dataset...")
    raw_dataset = load_div2k_dataset()
    total_samples = sum(1 for _ in raw_dataset)
    print(f"Total samples: {total_samples}")

    train_size = int(total_samples * 0.9)
    train_ds = raw_dataset.take(train_size)
    val_ds = raw_dataset.skip(train_size)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {total_samples - train_size}")

    train_dataset = prepare_dataset(train_ds, augment_data=True)
    val_dataset = prepare_dataset(val_ds, augment_data=False)

    print("Building model...")
    model = build_unet_denoising_model()
    model.compile(
        optimizer=Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.001, EPOCHS * train_size // BATCH_SIZE)),
        loss=combined_loss,
        metrics=[psnr_metric]
    )
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint('denoising_model_best.keras', monitor='val_psnr_metric', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    model.save('denoising_model_final.keras')

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr_metric'], label='Training PSNR')
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Test on validation set
    total_psnr = 0
    count = 0
    for noisy_batch, clean_batch in val_dataset.take(5):
        predictions = model.predict(noisy_batch)
        for i in range(len(noisy_batch)):
            psnr_val = peak_signal_noise_ratio(clean_batch[i].numpy(), predictions[i], data_range=1.0)
            total_psnr += psnr_val
            count += 1

    avg_psnr = total_psnr / count if count > 0 else 0
    print(f"Average PSNR on validation set: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    main()