# Audio Classification Project

## Overview

This project focuses on classifying audio files into 10 distinct classes. The approach involves converting audio signals into mel-spectrograms and then using Convolutional Neural Network (CNN) models built with TensorFlow/Keras to perform the classification. The project includes different iterations of model architectures and data augmentation techniques, documented in separate Jupyter Notebooks.

## Notebook Versions

The project contains a few iterations, with `audio-project-version6.ipynb` representing the latest and best-performing version.

### 📄 `audio-project-version6.ipynb` (Recommended)

This notebook implements an audio classification model with the following key features:
* **Input Features**: Mel-spectrograms (256x256, 1 channel, 256 mel bands) generated from audio files sampled at 44.1kHz.
* **Model Architecture**: A custom CNN utilizing:
    * Convolutional layers (Conv2D) followed by Batch Normalization and LeakyReLU activation.
    * Separable Convolutional layers (SeparableConv2D) for efficiency.
    * MaxPooling2D layers for down-sampling.
    * Dropout layers for regularization.
    * GlobalAveragePooling2D before the dense layers.
    * Dense layers with L2 kernel regularization, Batch Normalization, and LeakyReLU.
    * Softmax activation for the output layer (10 classes).
* **Data Augmentation**: Image augmentation techniques (rotation, width/height shift, zoom) are applied to the generated mel-spectrograms during training using `ImageDataGenerator`.
* **Training**:
    * Optimizer: Adam (learning_rate=0.001).
    * Loss Function: Categorical Crossentropy.
    * Callbacks: ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard.
    * Class weights are computed and used to handle class imbalance.
* **Results**: Achieved a **Test Accuracy of approximately 70.73%**.

### 📄 `audio-project-version5.ipynb` (and `audio-project.ipynb`)

This notebook represents an earlier iteration with a different custom CNN architecture:
* **Input Features**: Similar mel-spectrograms (256x256, 1 channel) as version 6.
* **Model Architecture**:
    * Multiple blocks of Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D.
    * Followed by Flatten -> Dense layers with Dropout.
* **Data Augmentation**:
    * **Audio Augmentation**: Applied directly to audio signals (add noise, time shift, pitch shift, time stretch) before spectrogram generation.
    * **Image Augmentation**: Applied to mel-spectrograms using `ImageDataGenerator` (rotation, width/height shift, zoom).
* **Training**: Similar setup to version 6 regarding optimizer, loss, and callbacks.
* **Results**: Achieved a **Test Accuracy of approximately 45.12%**.
    * This version also contains an experimental cell (cell 5) for a memory-efficient training approach using `EfficientNetB0`, which yielded lower accuracy in its initial configuration.

## Dataset

* The project uses the **Yildiz Teknik Proje 1 Train Dataset** (based on file paths in the notebooks, e.g., `/kaggle/input/yldz-teknik-proje-1-train-dataset/`).
* The dataset consists of audio files belonging to 10 classes.
* File names and labels are provided in `Train.csv` and `Test_Public.csv`.

## Methodology

1.  **Audio Preprocessing**:
    * Audio files are loaded using `librosa`.
    * They are resampled to a fixed sample rate (44100 Hz).
    * (In version 5) Audio augmentation techniques like adding noise, time shifting, pitch shifting, and time stretching are applied.
2.  **Feature Extraction**:
    * Mel-spectrograms are generated from the audio signals (`librosa.feature.melspectrogram`). Parameters include `n_mels=256` and `fmax=fixed_sample_rate // 2`.
    * Spectrograms are converted to dB scale (`librosa.amplitude_to_db`).
    * The resulting spectrograms are resized to a fixed image size (e.g., 256x256) and normalized.
3.  **Image Augmentation (for Spectrograms)**:
    * `ImageDataGenerator` is used to apply on-the-fly image augmentations like rotation, shifting, and zooming to the spectrograms during model training.
4.  **Model Training**:
    * CNN models are defined and compiled using TensorFlow/Keras.
    * Training involves using class weights to address potential imbalance in the dataset.
    * Callbacks such as `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` are used to manage the training process and save the best model.
5.  **Evaluation**:
    * The trained models are evaluated on a public test set to measure performance using metrics like loss and accuracy.
    * The best performing model weights are saved and loaded for testing (model `kerim_best.keras` or `kerim.keras`).

## Key Libraries & Frameworks

* Python 3
* TensorFlow / Keras
* Librosa (for audio processing and feature extraction)
* Pandas (for data handling from CSV files)
* NumPy (for numerical operations)
* Pillow (PIL) (for image manipulation)
* Matplotlib & Seaborn (for plotting, though plots are mainly for accuracy/loss curves)
* Scikit-learn (for `train_test_split` and `compute_class_weight`)
* Gdown (used in notebooks to download pre-trained model weights for testing)

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Install Dependencies:**
    Ensure you have Python 3 installed. Key dependencies can be installed via pip:
    ```bash
    pip install tensorflow pandas numpy librosa Pillow scikit-learn matplotlib seaborn gdown resampy
    ```
3.  **Dataset:**
    * The notebooks expect the dataset to be available, typically in a Kaggle input directory structure (e.g., `/kaggle/input/yldz-teknik-proje-1-train-dataset/`).
    * You will need to download the "Yildiz Teknik Proje 1 Train Dataset" and place the `Train/`, `Test_Public/`, `Train.csv`, and `Test_Public.csv` files in the appropriate location or update the paths in the notebooks.
4.  **Run Notebooks:**
    * Open and run the Jupyter Notebooks (`audio-project-version5.ipynb` or `audio-project-version6.ipynb`) using Jupyter Lab or Jupyter Notebook.
    * It is recommended to run `audio-project-version6.ipynb` for the best results.

## Results Summary

* **Version 5 Model**: Achieved a test accuracy of approximately **45.1%**.
* **Version 6 Model**: Showed significant improvement, achieving a test accuracy of approximately **70.7%**. This version features a more advanced CNN architecture and is the recommended model.

---

Feel free to modify any part to better suit your project's specifics!
