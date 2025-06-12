
# 🫀 ECG Signal Processing Techniques for ML/DL

## 1. **Preprocessing & Cleaning**

### 1.1 Signal Normalization

* Min-Max scaling or Z-score normalization to standardize amplitude across recordings.

### 1.2 Baseline Wander Correction

* Removes low-frequency drift using high-pass filtering or polynomial detrending.

### 1.3 Noise & Artifact Removal

* Motion artifact reduction, muscle noise suppression, and powerline interference removal using:

  * Filtering (Notch, Bandpass, FIR/IIR)
  * Wavelet denoising
  * Autoencoder-based denoising

### 1.4 Handling Missing Data

* Techniques like interpolation, forward/backward filling, or model-based reconstruction.

---

## 2. **Temporal Processing**

### 2.1 Resampling / Interpolation

* Uniform sampling (e.g., from 250 Hz to 360 Hz) for consistent input shape.

### 2.2 Windowing / Fixed-Length Segmentation

* Sliding windows or beat-based windows (with/without overlap) for feeding into models.

### 2.3 Beat Segmentation

* Extracting individual heartbeats using R-peak detection (e.g., Pan-Tompkins).

### 2.4 Morphological Alignment

* Aligning ECG waveforms (especially QRS complexes) using DTW or cross-correlation.

---

## 3. **Feature Engineering**

### 3.1 R-Peak Detection

* Identifies R-peaks to segment beats or compute HRV metrics.

### 3.2 Time-Domain Features

* RR intervals, QRS duration, PR/QT intervals, peak amplitudes.

### 3.3 Frequency-Domain Features

* Fourier Transform, Power Spectral Density (PSD).

### 3.4 Time-Frequency Features

* Wavelet transforms, Short-Time Fourier Transform (STFT).

---

## 4. **Dimensionality Reduction & Channel Handling**

### 4.1 Lead Selection / Fusion

* Selecting the most informative leads (e.g., from 12-lead ECG) or fusing them via PCA or channel averaging.

### 4.2 Principal Component Analysis (PCA)

* Reduces redundancy across channels or features.

### 4.3 Independent Component Analysis (ICA)

* Separates mixed signal sources (e.g., noise vs actual cardiac activity).

---

## 5. **Augmentation & Synthesis**

### 5.1 Data Augmentation

* Adding Gaussian noise, time scaling, amplitude jittering, or simulating artifacts.

### 5.2 Synthetic ECG Generation

* GANs or rule-based synthetic signal generation for class balancing.

---

## 6. **Advanced Signal Denoising**

### 6.1 Wavelet-based Denoising

* Decomposes signals into frequency bands and removes noise components.

### 6.2 Empirical Mode Decomposition (EMD)

* Decomposes signals into Intrinsic Mode Functions (IMFs).

### 6.3 Variational Mode Decomposition (VMD)

* Improved version of EMD for separating signal modes.

### 6.4 Non-local Means Filtering

* Averaging similar signal patches across the dataset for noise suppression.

### 6.5 Kalman Filtering

* Predictive filtering to remove dynamic noise.

---

## ✅ Bonus Utility

* **Signal Padding / Cropping:** To ensure consistent input sizes for DL models.
* **Label Alignment:** Ensuring that signal segments are synchronized with diagnostic labels or annotations.

