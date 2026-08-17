import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# --- Configuration ---
PROCESSED_DATA_DIR = 'processed_master_dataset'
REAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'train', 'real')
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LATENT_DIM = 128  # Dimension of the compressed "latent" space

# --- 1. The VAE Model ---
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # Unpack the data. The generator yields (images, images).
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            
            # --- The VAE's special combined loss ---
            reconstruction_loss = tf.reduce_mean(tf.square(y - reconstruction)) * (IMG_SIZE[0] * IMG_SIZE[1])
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

# --- Sampling Layer (The key to a VAE) ---
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Build Encoder and Decoder ---
def build_vae_components(input_shape, latent_dim):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 32 * 64, activation="relu")(latent_inputs)
    x = Reshape((32, 32, 64))(x)
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    
    return encoder, decoder

# --- Main Training Logic ---
if __name__ == "__main__":
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        classes=['real'],
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='input' # Use 'input' to make it yield (images, images)
    )

    encoder, decoder = build_vae_components((IMG_SIZE[0], IMG_SIZE[1], 3), LATENT_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    print("\nStarting VAE training...")
    vae.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(train_generator))

    print("\n✅ VAE training complete!")
    encoder.save('face_vae_encoder.h5')
    decoder.save('face_vae_decoder.h5')
    print("Encoder and Decoder models saved.")