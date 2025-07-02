### What is Feature Extraction in ECG?

Imagine an ECG (electrocardiogram) as a story written by your heart. It’s a wiggly line on a graph that shows how your heart’s electrical signals change over time. But this line is packed with information, and we need to pull out specific pieces—called **features**—to understand what’s happening, like whether the heart is beating normally or if there’s a problem.

**Feature extraction** is the process of finding and measuring these important pieces of information from the ECG signal. These features are like puzzle pieces that we can feed into a computer (using machine learning or deep learning) to detect heart diseases, classify heartbeats, or predict health issues. There are three main types of features we’ll explore:

1. **Time-Domain Features**: These focus on the timing and shape of the ECG signal in the time dimension (like how long a heartbeat takes or how tall a wave is).
2. **Frequency-Domain Features**: These look at the “vibrations” or frequencies in the ECG signal (like how fast or slow the signal oscillates).
3. **Time-Frequency Domain Features**: These combine time and frequency to see how the signal’s frequencies change over time.

Let’s break each one down with explanations and examples, so you can understand them step by step.

---

## 6.1 Time-Domain Features

### What Are Time-Domain Features?

Time-domain features are measurements we take directly from the ECG signal as it appears on a graph over time. Think of the ECG as a line on a timeline: we’re looking at how long things take, how high or low the line goes, or how the shape of the line looks. These features are easy to understand because they’re based on the raw signal, without transforming it into something else.

An ECG signal has specific parts, like waves and intervals, that tell us about the heart’s activity. The main components are:

- **P Wave**: The small bump before the big spike, showing the atria (upper heart chambers) contracting.
- **QRS Complex**: The big spike (Q is a dip, R is a peak, S is another dip), showing the ventricles (lower heart chambers) contracting.
- **T Wave**: The wave after the QRS, showing the ventricles relaxing.

Time-domain features measure things like the time between these waves, their heights, or their shapes. Let’s look at some key time-domain features, starting with the ones you mentioned: **RR Interval** and **PR Interval**.

### Key Time-Domain Features

Here’s a detailed list of time-domain features you’ll often encounter in ECG analysis:

1. **RR Interval**: The time between two consecutive R peaks (the tallest spike in the QRS complex). It tells us the heart rate. A shorter RR interval means a faster heart rate, and a longer one means a slower heart rate.
2. **PR Interval**: The time from the start of the P wave to the start of the QRS complex. It shows how long it takes for the electrical signal to travel from the atria to the ventricles.
3. **QRS Duration**: The time from the start of the Q wave to the end of the S wave. It shows how long the ventricles take to contract.
4. **QT Interval**: The time from the start of the Q wave to the end of the T wave. It represents the total time for ventricular contraction and relaxation.
5. **P Wave Duration**: The time from the start to the end of the P wave, showing how long the atria take to contract.
6. **T Wave Duration**: The time from the start to the end of the T wave, showing ventricular relaxation time.
7. **P Wave Amplitude**: The height of the P wave, indicating the strength of atrial contraction.
8. **R Wave Amplitude**: The height of the R peak, indicating the strength of ventricular contraction.
9. **Heart Rate Variability (HRV)**: The variation in RR intervals over time, which can show how adaptable the heart is to stress or rest.
10. **ST Segment Elevation/Depression**: The level of the line between the S wave and T wave, which can indicate heart issues like ischemia (lack of blood flow).

### Why Are These Important?

Time-domain features are like the vital signs of an ECG. For example:
- A long PR interval might suggest a delay in the heart’s electrical conduction (like a first-degree heart block).
- A wide QRS duration could indicate a problem with ventricular contraction, like a bundle branch block.
- HRV can tell us about the autonomic nervous system’s control over the heart, which is useful for detecting stress or heart failure.

These features are simple to compute and are often used in traditional ECG analysis by doctors. In machine learning, we extract these features to train models to automatically detect patterns that humans might miss.

### End-to-End Example: Extracting RR and PR Intervals

Let’s walk through a simple example of how to extract **RR Interval** and **PR Interval** from an ECG signal using Python. We’ll use the **Neurokit2** library, which is beginner-friendly for ECG processing.

#### Step 1: Install Required Tools
You need Python and some libraries. If you don’t have them, install them using:
```bash
pip install neurokit2 numpy matplotlib
```

#### Step 2: Get ECG Data
We’ll use a sample ECG from the **Physionet MIT-BIH Arrhythmia Database**, but for simplicity, let’s assume we have a small ECG signal (in millivolts) recorded at 360 Hz (360 samples per second).

#### Step 3: Python Code to Extract RR and PR Intervals

```python

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

# Simulated ECG signal (replace with real data from Physionet if available)
# Let's create a fake ECG signal for demonstration (1 second, 360 Hz)
sampling_rate = 360  # Hz
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))  # Fake ECG with noise

# Process the ECG signal using Neurokit2
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

# Find peaks (R-peaks and others)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract R-peaks and P-peaks
r_peaks = info['ECG_R_Peaks']
p_peaks = info['ECG_P_Peaks']

# Calculate RR Intervals (time between consecutive R-peaks)
rr_intervals = np.diff(r_peaks) / sampling_rate  # Convert to seconds
print("RR Intervals (seconds):", rr_intervals)

# Calculate PR Intervals (time from P-peak to R-peak)
pr_intervals = []
for i in range(len(p_peaks)):
    if i < len(r_peaks):  # Ensure we have corresponding R-peak
        pr_interval = (r_peaks[i] - p_peaks[i]) / sampling_rate  # Convert to seconds
        pr_intervals.append(pr_interval)
print("PR Intervals (seconds):", pr_intervals)

# Plot the ECG signal with R and P peaks
plt.plot(ecg_cleaned, label="ECG Signal")
plt.plot(r_peaks, ecg_cleaned[r_peaks], "ro", label="R Peaks")
plt.plot(p_peaks, ecg_cleaned[p_peaks], "go", label="P Peaks")
plt.legend()
plt.title("ECG Signal with R and P Peaks")
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.show()

```

#### Explanation of the Code
1. **Import Libraries**: We use `neurokit2` for ECG processing, `numpy` for calculations, and `matplotlib` for plotting.
2. **Simulated ECG**: Since we’re keeping it simple, we create a fake ECG signal (a sine wave with noise). In a real project, you’d load data from a database like MIT-BIH.
3. **Clean the Signal**: `nk.ecg_clean` removes noise to make peak detection easier.
4. **Find Peaks**: `nk.ecg_process` detects R-peaks (the tallest spikes) and P-peaks (smaller bumps before QRS).
5. **Calculate RR Intervals**: We compute the time between consecutive R-peaks by taking the difference (`np.diff`) and dividing by the sampling rate to get seconds.
6. **Calculate PR Intervals**: We find the time from each P-peak to the corresponding R-peak.
7. **Visualize**: We plot the ECG with marked R and P peaks to see where they are.

#### What You’ll See
When you run this code, it will:
- Print the RR intervals (e.g., time between heartbeats in seconds).
- Print the PR intervals (e.g., time from atrial to ventricular activation).
- Show a graph of the ECG signal with red dots for R-peaks and green dots for P-peaks.

This is a simple way to extract time-domain features. In a real project, you’d use these features to train a machine learning model to classify heartbeats (e.g., normal vs. abnormal).

---

## 6.2 Frequency-Domain Features

### What Are Frequency-Domain Features?

Frequency-domain features look at the ECG signal in terms of its **frequencies**—how fast or slow the signal “vibrates.” Instead of measuring time or amplitude directly, we transform the ECG signal into a frequency spectrum to see which frequencies are present and how strong they are. This is useful because some heart conditions (like arrhythmias) show up as specific frequency patterns.

Think of the ECG as music: time-domain features are like the notes you hear, while frequency-domain features are like the pitch or tone of the music. We use mathematical tools like the **Fourier Transform** to convert the ECG from time to frequency.

### Key Frequency-Domain Features

Here are some important frequency-domain features for ECG:

1. **Power Spectral Density (PSD)**: Shows how the signal’s power (energy) is distributed across different frequencies.
2. **Low-Frequency (LF) Power**: Energy in the 0.04–0.15 Hz range, often linked to sympathetic nervous system activity.
3. **High-Frequency (HF) Power**: Energy in the 0.15–0.4 Hz range, linked to parasympathetic nervous system activity (e.g., breathing).
4. **LF/HF Ratio**: The balance between sympathetic and parasympathetic activity, used in heart rate variability (HRV) analysis.
5. **Total Power**: The total energy across all frequencies, indicating overall signal strength.
6. **Peak Frequency**: The frequency with the highest power, which might correspond to the heart rate.
7. **Spectral Entropy**: Measures the randomness or complexity of the frequency spectrum.
8. **Dominant Frequency**: The most prominent frequency in the signal, often related to the heart’s rhythm.
9. **Spectral Centroid**: The “center of mass” of the frequency spectrum, indicating where most energy is concentrated.
10. **Frequency Band Energy**: Energy in specific frequency bands (e.g., very low frequency, 0–0.04 Hz).

### Why Are These Important?

Frequency-domain features are great for analyzing **heart rate variability (HRV)** and detecting patterns that aren’t obvious in the time domain. For example:
- A high LF/HF ratio might indicate stress or a heart condition.
- Abnormal PSD patterns can help detect arrhythmias like atrial fibrillation, where the heart’s rhythm becomes irregular.

These features are often used in machine learning to classify heart conditions or monitor stress levels.

### End-to-End Example: Calculating Power Spectral Density (PSD)

Let’s compute the **Power Spectral Density** of an ECG signal using Python. We’ll use the `scipy` library to perform a Fourier Transform and calculate the PSD.

#### Step 1: Install Required Tools
Install the necessary libraries:
```bash
pip install scipy numpy matplotlib
```

#### Step 2: Python Code to Compute PSD

```python

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Simulated ECG signal (1 second, 360 Hz)
sampling_rate = 360  # Hz
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))  # Fake ECG with noise

# Calculate Power Spectral Density (PSD) using Welch's method
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Print key frequency-domain features
print("Frequencies (Hz):", frequencies[:10])  # First 10 frequencies for brevity
print("PSD Values:", psd[:10])  # Corresponding PSD values

# Plot the PSD
plt.plot(frequencies, psd)
plt.title("Power Spectral Density of ECG Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (mV²/Hz)")
plt.grid(True)
plt.show()

# Calculate LF and HF power (integrate PSD over frequency bands)
lf_band = (0.04, 0.15)  # Low-frequency range
hf_band = (0.15, 0.4)   # High-frequency range

lf_power = np.trapz(psd[(frequencies >= lf_band[0]) & (frequencies <= lf_band[1])], 
                     frequencies[(frequencies >= lf_band[0]) & (frequencies <= lf_band[1])])
hf_power = np.trapz(psd[(frequencies >= hf_band[0]) & (frequencies <= hf_band[1])], 
                     frequencies[(frequencies >= hf_band[0]) & (frequencies <= hf_band[1])])

print("Low-Frequency Power (0.04–0.15 Hz):", lf_power)
print("High-Frequency Power (0.15–0.4 Hz):", hf_power)
print("LF/HF Ratio:", lf_power / hf_power)

```

#### Explanation of the Code
1. **Simulated ECG**: We create a fake ECG signal (a sine wave with noise) at 360 Hz.
2. **Welch’s Method**: `signal.welch` computes the PSD, which estimates how power is distributed across frequencies. The `nperseg=256` parameter sets the window size for the calculation.
3. **Plot PSD**: We plot the frequencies (x-axis) vs. PSD values (y-axis) to see the frequency content of the signal.
4. **LF and HF Power**: We integrate (sum) the PSD over the low-frequency (0.04–0.15 Hz) and high-frequency (0.15–0.4 Hz) bands using `np.trapz` to calculate their power.
5. **LF/HF Ratio**: We compute the ratio to assess the balance between sympathetic and parasympathetic activity.

#### What You’ll See
When you run this code, it will:
- Show a graph of the PSD, with peaks indicating dominant frequencies in the ECG.
- Print the LF power, HF power, and LF/HF ratio, which are useful for HRV analysis.

In a real project, you’d use these features to train a machine learning model to detect stress or arrhythmias based on frequency patterns.

---

## 6.3 Time-Frequency Domain Features

### What Are Time-Frequency Domain Features?

Time-frequency domain features combine the best of both worlds: they show how the frequencies in an ECG signal change over time. Unlike frequency-domain features, which give a single snapshot of all frequencies, time-frequency features let us see how the signal’s “pitch” evolves as the heart beats. This is crucial for ECGs because heart signals are **non-stationary**—their patterns change over time, especially in conditions like arrhythmias.

Think of it like watching a movie instead of a single photo: time-frequency analysis shows how the ECG’s frequencies “dance” over time.

### Key Time-Frequency Domain Features

Here are some important time-frequency features:

1. **Short-Time Fourier Transform (STFT)**: Divides the signal into short windows and computes the Fourier Transform for each, showing how frequencies change over time.
2. **Continuous Wavelet Transform (CWT)**: Uses wavelets (small wave-like functions) to analyze frequencies at different scales, great for non-stationary signals like ECG.
3. **Spectrogram**: The squared magnitude of the STFT, showing frequency intensity over time.
4. **Scalogram**: The squared magnitude of the CWT, showing how wavelet coefficients vary over time and scale.
5. **Wavelet Energy**: The energy in specific wavelet scales, indicating the strength of certain frequency bands.
6. **Instantaneous Frequency**: The frequency at a specific time, derived from time-frequency methods.
7. **Time-Frequency Entropy**: Measures the complexity or randomness of the signal’s frequency content over time.
8. **Morlet Wavelet Coefficients**: Specific wavelet features used in CWT, good for ECG morphology.
9. **Wigner-Ville Distribution**: A high-resolution time-frequency method, though sensitive to noise.
10. **Chirplet Transform**: Analyzes frequency changes that vary linearly over time, useful for specific ECG patterns.

### Why Are These Important?

Time-frequency features are powerful because they capture dynamic changes in the ECG. For example:
- In atrial fibrillation, the frequencies of the ECG signal change irregularly, and time-frequency analysis can spot this.
- Wavelet transforms are great for detecting sudden changes, like premature ventricular contractions (PVCs).
- Spectrograms can show how the heart’s rhythm evolves during stress or exercise.

These features are often used in deep learning models, where the time-frequency representation (like a spectrogram) is treated like an image for a convolutional neural network (CNN).

### End-to-End Example: Computing Short-Time Fourier Transform (STFT)

Let’s compute the **STFT** of an ECG signal and visualize it as a spectrogram using Python. This will show how frequencies change over time.

#### Step 1: Install Required Tools
Install the necessary libraries:
```bash
pip install scipy numpy matplotlib
```

#### Step 2: Python Code to Compute STFT

```python

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Simulated ECG signal (2 seconds, 360 Hz)
sampling_rate = 360  # Hz
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute Short-Time Fourier Transform (STFT)
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)

# Plot the spectrogram (magnitude of STFT)
plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
plt.title("Spectrogram of ECG Signal (STFT)")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label="Amplitude")
plt.show()

# Extract time-frequency feature: Total energy in a specific frequency band (e.g., 0–10 Hz)
freq_band = (0, 10)
freq_mask = (frequencies >= freq_band[0]) & (frequencies <= freq_band[1])
band_energy = np.sum(np.abs(Zxx[freq_mask, :])**2)
print("Total Energy in 0–10 Hz Band:", band_energy)

```

#### Explanation of the Code
1. **Simulated ECG**: We create a fake ECG signal with two frequencies (2 Hz and 5 Hz) plus noise to mimic a real signal.
2. **STFT**: `signal.stft` computes the Short-Time Fourier Transform, dividing the signal into short windows (128 samples) and calculating the frequency content for each.
3. **Spectrogram**: We plot the magnitude of the STFT (`np.abs(Zxx)`) as a heatmap, where the x-axis is time, the y-axis is frequency, and the color shows amplitude.
4. **Band Energy**: We calculate the total energy in the 0–10 Hz band by summing the squared magnitudes of the STFT in that frequency range.

#### What You’ll See
When you run this code, it will:
- Show a spectrogram, where brighter colors indicate stronger frequencies at specific times.
- Print the total energy in the 0–10 Hz band, which could be used as a feature for machine learning.

In a real project, you might feed the spectrogram into a CNN to classify heart conditions or use wavelet coefficients for more detailed analysis.

---

### Putting It All Together

Now that you’ve learned about **time-domain**, **frequency-domain**, and **time-frequency domain features**, here’s how they fit into ECG research:

1. **Time-Domain Features** (like RR and PR intervals) are simple and directly relate to the heart’s electrical activity. They’re great for traditional analysis and basic machine learning models.
2. **Frequency-Domain Features** (like PSD) reveal the hidden “vibrations” in the ECG, useful for HRV analysis and detecting rhythmic patterns.
3. **Time-Frequency Domain Features** (like STFT) capture how the ECG changes over time, perfect for dynamic conditions like arrhythmias.

In a PhD project, you might:
- Extract time-domain features (e.g., RR intervals) and frequency-domain features (e.g., LF/HF ratio) to train a simple machine learning model like a Random Forest.
- Use time-frequency features (e.g., spectrograms) as input to a deep learning model like a CNN to detect complex patterns.
- Combine all three types of features to improve model accuracy.

### Tips for Learning More
- **Practice with Real Data**: Download ECG data from Physionet (e.g., MIT-BIH Arrhythmia Database) and try extracting these features.
- **Use Libraries**: Neurokit2, Biosppy, and SciPy are great for beginners. They handle complex math for you.
- **Visualize Everything**: Plot your signals, peaks, and spectrograms to understand what’s happening.
- **Start Small**: Begin with time-domain features (they’re the easiest), then move to frequency and time-frequency features as you get comfortable.
