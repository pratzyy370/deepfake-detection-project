# Deepfake Detection System

A deep learning-based deepfake detection system developed to identify manipulated facial images and videos using supervised and unsupervised machine learning approaches. The project evaluates multiple neural network architectures and compares their effectiveness in detecting synthetic media.

This work was completed as part of a research study investigating the strengths and limitations of different deepfake detection techniques using the Celeb-DF-v2 dataset.

---

## Research Paper

The complete research paper associated with this project is available in the `research-paper` directory.

The paper documents:

* Dataset preparation
* Model architectures
* Experimental methodology
* Evaluation metrics
* Comparative analysis of results
* Limitations and future work

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Flask
* HTML Templates
* Celeb-DF-v2 Dataset

---

## Features

* Deepfake image detection
* Deepfake video analysis
* Face extraction and preprocessing
* Supervised classification using XceptionNet
* Autoencoder-based anomaly detection
* Variational Autoencoder (VAE) based detection
* Web-based interface for inference
* Model evaluation and performance comparison

---

## Project Structure

```text
Deepfake-Detection/
│
├── app.py
├── train.py
├── evaluate.py
├── train_autoencoder.py
├── train_vae.py
├── prepare_dataset.py
├── convert_videos.py
├── split_dataset.py
├── templates/
├── screenshots/
├── research-paper/
│   └── Deepfake_Detection_Research_Paper.pdf
└── README.md
```

---

## Dataset

The models were trained and evaluated using the Celeb-DF-v2 dataset, a benchmark dataset containing authentic and manipulated facial videos commonly used for deepfake detection research.

Dataset preprocessing included:

* Video frame extraction
* Face detection
* Face cropping
* Image normalization
* Dataset splitting for training, validation and testing

---

## Models Evaluated

### Model A: XceptionNet Classifier

A supervised deep learning classifier trained to distinguish between authentic and manipulated facial images.

### Model B: Autoencoder Anomaly Detector

An unsupervised reconstruction-based model that attempts to identify anomalies through reconstruction error.

### Model C: Variational Autoencoder (VAE)

A latent-space anomaly detection approach designed to separate authentic and manipulated images using learned feature representations.

---

## Results

| Model                         | Accuracy | Precision | Recall | F1 Score |
| ----------------------------- | -------- | --------- | ------ | -------- |
| XceptionNet Classifier        | 88.13%   | 0.86      | 0.91   | 0.89     |
| Autoencoder                   | 50.00%   | 0.00      | 0.00   | 0.00     |
| Variational Autoencoder (VAE) | 50.18%   | 0.50      | 1.00   | 0.67     |

---

## Key Findings

The XceptionNet classifier achieved the strongest performance among all evaluated models, reaching an accuracy of 88.13% and an F1 score of 0.89.

The Autoencoder failed to effectively distinguish between authentic and manipulated images because reconstruction errors remained similar for both classes.

The Variational Autoencoder achieved perfect recall for fake images but incorrectly classified a large number of authentic images as manipulated, resulting in poor overall accuracy despite detecting all deepfake samples.

These findings demonstrate that supervised deep learning approaches remain significantly more effective than reconstruction-based anomaly detection methods for deepfake detection on the evaluated dataset.

---

## Screenshots 
![Sample Detection](https://github.com/pratzyy370/deepfake-detection-project/blob/main/screenshots/Screenshot%202025-12-05%20225539.png?raw=true)

## How to Run

1. Clone the repository

```bash
git clone https://github.com/pratzyy370/deepfake-detection-project.git
cd deepfake-detection-project
```

2. Create a virtual environment

```bash
python -m venv .venv
```

3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

5. Run the application

```bash
python app.py
```

---

## Future Improvements

* Improve detection accuracy on unseen deepfake techniques
* Support real-time webcam detection
* Expand dataset diversity
* Improve video-level inference
* Deploy as a cloud-hosted web application
* Explore transformer-based architectures

---

## Author

Prathyush Rao M


