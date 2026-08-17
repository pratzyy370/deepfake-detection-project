import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
# --- NEW: Import the Dropout layer ---
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- Configuration ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
DATA_DIR = 'processed_master_dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')

# --- Data Preparation with More Aggressive Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,      # Increased rotation
    zoom_range=0.2,         # Increased zoom
    width_shift_range=0.1,  # --- NEW: Randomly shift images horizontally ---
    height_shift_range=0.1, # --- NEW: Randomly shift images vertically ---
    shear_range=0.1         # --- NEW: Shear transformations ---
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- Build the Model with Dropout Layers ---
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in base_model.layers[:-10]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # --- NEW: Dropout layer to prevent overfitting ---
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # --- NEW: Another Dropout layer ---
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Compile and Train ---
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("Starting Classifier training with anti-overfitting techniques...")
history = model.fit(
    train_generator,
    epochs=15, # Increased epochs slightly to give it more time to learn from harder data
    validation_data=validation_generator
)

# --- Save the Final Model ---
model.save('classifier_model_robust.h5')
print("\n✅ Final robust classifier model training complete! Saved as classifier_model_robust.h5")