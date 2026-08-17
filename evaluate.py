import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_fscore_support, accuracy_score

# --- NEW: Define the Custom Sampling Layer from train_vae.py ---
# We must include this so Keras knows what a "Sampling" layer is when loading the model.
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Configuration ---
DATA_DIR = 'processed_master_dataset'
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
IMG_SIZE_CLASSIFIER = (299, 299)
IMG_SIZE_AE_VAE = (128, 128)
BATCH_SIZE = 32

# --- Load All Models ---
print("Loading all three models...")
classifier = load_model('classifier_model_robust.h5')
autoencoder = load_model('face_autoencoder.h5')
# --- UPDATED: Tell load_model about our custom layer ---
vae_encoder = load_model('face_vae_encoder.h5', custom_objects={'Sampling': Sampling})
vae_decoder = load_model('face_vae_decoder.h5')
print("Models loaded successfully.")

# --- Data Generators ---
datagen = ImageDataGenerator(rescale=1./255)

def get_full_dataset(path, img_size):
    generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    images = []
    labels = []
    for i in range(len(generator)):
        imgs, lbls = generator[i]
        images.extend(imgs)
        labels.extend(lbls)
    return np.array(images), np.array(labels)

# --- 1. Evaluate Supervised Classifier (Model A) ---
print("\n--- [Model A] Evaluating Supervised Classifier ---")
test_images_classifier, y_true_classifier = get_full_dataset(TEST_DIR, IMG_SIZE_CLASSIFIER)
y_pred_classifier_probs = classifier.predict(test_images_classifier)
y_pred_classifier = (y_pred_classifier_probs > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_true_classifier, y_pred_classifier)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_classifier, y_pred_classifier, average='binary', pos_label=0)
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f"  Precision (for 'fake'): {precision:.2f}")
print(f"  Recall (for 'fake'): {recall:.2f}")
print(f"  F1-Score (for 'fake'): {f1:.2f}")

# --- Function to Evaluate Anomaly Detectors ---
def evaluate_anomaly_detector(model_name, model, val_images, val_labels, test_images, test_labels):
    print(f"\n--- [{model_name}] Evaluating Anomaly Detector ---")
    reconstructed_val = model.predict(val_images)
    val_errors = np.mean(np.square(val_images - reconstructed_val), axis=(1, 2, 3))
    fpr, tpr, thresholds = roc_curve(1 - val_labels, val_errors) 
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"  Optimal anomaly threshold found: {optimal_threshold:.6f}")

    reconstructed_test = model.predict(test_images)
    test_errors = np.mean(np.square(test_images - reconstructed_test), axis=(1, 2, 3))
    y_pred = (test_errors < optimal_threshold).astype(int)
    accuracy_ae = accuracy_score(test_labels, y_pred)
    precision_ae, recall_ae, f1_ae, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary', pos_label=0)
    print(f"  Accuracy: {accuracy_ae * 100:.2f}%")
    print(f"  Precision (for 'fake'): {precision_ae:.2f}")
    print(f"  Recall (for 'fake'): {recall_ae:.2f}")
    print(f"  F1-Score (for 'fake'): {f1_ae:.2f}")

# --- 2. Evaluate Autoencoder (Model B) ---
val_images_ae, val_labels_ae = get_full_dataset(VALIDATION_DIR, IMG_SIZE_AE_VAE)
test_images_ae, test_labels_ae = get_full_dataset(TEST_DIR, IMG_SIZE_AE_VAE)
evaluate_anomaly_detector("Model B - Autoencoder", autoencoder, val_images_ae, val_labels_ae, test_images_ae, test_labels_ae)

# --- 3. Evaluate VAE (Model C) ---
class VAE_Wrapper(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE_Wrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def predict(self, data):
        z_mean, z_log_var, z = self.encoder.predict(data)
        return self.decoder.predict(z)

vae_full = VAE_Wrapper(vae_encoder, vae_decoder)
evaluate_anomaly_detector("Model C - VAE", vae_full, val_images_ae, val_labels_ae, test_images_ae, test_labels_ae)
    
print("\n✅ Evaluation Complete.")