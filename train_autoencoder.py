import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
PROCESSED_DATA_DIR = 'processed_master_dataset'
REAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'train', 'real')
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(inputs, decoded)
    return autoencoder

# --- NEW: Generator Wrapper Function ---
# This wrapper takes the output of the original generator and yields it
# in the (input, target) format that model.fit() expects for an autoencoder.
def autoencoder_generator(generator):
    while True:
        # The original generator yields only the images (x)
        x = next(generator)
        # We yield the images as both input (x) and target (y)
        yield (x, x)

if __name__ == "__main__":
    if not os.path.exists(REAL_DATA_DIR):
        print(f"❌ Error: Real data directory not found at '{REAL_DATA_DIR}'")
        exit()

    datagen = ImageDataGenerator(rescale=1./255)

    # This generator correctly loads the images
    image_generator = datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        classes=['real'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None
    )

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    autoencoder = build_autoencoder(input_shape)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Autoencoder Model Summary:")
    autoencoder.summary()
    
    print("\nStarting Autoencoder training (learning what 'real' looks like)...")
    
    # --- UPDATED: Use the new wrapper function here ---
    autoencoder.fit(
        autoencoder_generator(image_generator),
        epochs=EPOCHS,
        steps_per_epoch=len(image_generator)
    )

    autoencoder.save('face_autoencoder.h5')
    print("\n✅ Autoencoder training complete! Model saved as face_autoencoder.h5")