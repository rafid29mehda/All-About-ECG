## 6.1 Time-Domain Features: Extraction Techniques

Time-domain features are measurements taken directly from the ECG signal in the time dimension, focusing on timing, amplitude, or shape of the signal’s components (e.g., P wave, QRS complex, T wave). Below, I’ll describe the extraction techniques for each of the 10 time-domain features listed previously: **RR Interval**, **PR Interval**, **QRS Duration**, **QT Interval**, **P Wave Duration**, **T Wave Duration**, **P Wave Amplitude**, **R Wave Amplitude**, **Heart Rate Variability (HRV)**, and **ST Segment Elevation/Depression**.

### 1. RR Interval
**Description**: The time between consecutive R peaks (the tallest peak in the QRS complex), used to calculate heart rate.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal to remove noise (e.g., baseline wander, muscle artifacts).
- **Step 2**: Detect R peaks using a peak detection algorithm (e.g., Pan-Tompkins algorithm).
- **Step 3**: Compute the time difference between consecutive R peaks in seconds (divide sample difference by sampling rate).
- **Tools**: Libraries like `neurokit2` or `biosppy` automate R-peak detection.
- **Example**:
  - Input: ECG signal sampled at 360 Hz.
  - Process: Detect R peaks, calculate differences.
  - Output: RR intervals in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal (1 second, 360 Hz)
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract RR intervals
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate  # Convert to seconds
print("RR Intervals (seconds):", rr_intervals)
```

**Explanation**: The code uses `neurokit2` to clean the ECG and detect R peaks. The `np.diff` function calculates the time between consecutive R peaks, converted to seconds by dividing by the sampling rate.

---

### 2. PR Interval
**Description**: The time from the start of the P wave to the start of the QRS complex, indicating atrial-to-ventricular conduction time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect P wave onsets and QRS complex onsets (Q wave or R wave start).
- **Step 3**: Compute the time difference between P wave onset and QRS onset.
- **Tools**: `neurokit2` can detect P wave and QRS onsets.
- **Example**:
  - Input: ECG signal.
  - Process: Identify P wave start and QRS start, calculate difference.
  - Output: PR intervals in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract PR intervals
p_onsets = info['ECG_P_Onsets']
qrs_onsets = info['ECG_QRS_Onsets']
pr_intervals = [(qrs_onsets[i] - p_onsets[i]) / sampling_rate for i in range(min(len(p_onsets), len(qrs_onsets)))]
print("PR Intervals (seconds):", pr_intervals)
```

**Explanation**: The code detects P wave and QRS onsets using `neurokit2` and computes the time difference. If onsets are missing, you may need manual detection or better signal quality.

---

### 3. QRS Duration
**Description**: The time from the start of the Q wave to the end of the S wave, showing ventricular depolarization time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect QRS complex boundaries (Q wave start, S wave end).
- **Step 3**: Compute the time difference between Q and S points.
- **Tools**: `neurokit2` or manual peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify Q and S points, calculate duration.
  - Output: QRS durations in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract QRS durations
qrs_onsets = info['ECG_QRS_Onsets']
qrs_offsets = info['ECG_QRS_Offsets']
qrs_durations = [(qrs_offsets[i] - qrs_onsets[i]) / sampling_rate for i in range(min(len(qrs_onsets), len(qrs_offsets)))]
print("QRS Durations (seconds):", qrs_durations)
```

**Explanation**: The code identifies QRS complex boundaries and computes their duration. Accurate QRS detection requires a clean signal.

---

### 4. QT Interval
**Description**: The time from the start of the Q wave to the end of the T wave, representing ventricular depolarization and repolarization.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect Q wave onset and T wave offset.
- **Step 3**: Compute the time difference.
- **Tools**: `neurokit2` for automated detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify Q onset and T offset, calculate difference.
  - Output: QT intervals in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract QT intervals
qrs_onsets = info['ECG_QRS_Onsets']
t_offsets = info['ECG_T_Offsets']
qt_intervals = [(t_offsets[i] - qrs_onsets[i]) / sampling_rate for i in range(min(len(qrs_onsets), len(t_offsets)))]
print("QT Intervals (seconds):", qt_intervals)
```

**Explanation**: The code computes the time from QRS onset to T wave offset. T wave detection can be tricky due to its variability, so ensure high signal quality.

---

### 5. P Wave Duration
**Description**: The time from the start to the end of the P wave, showing atrial depolarization time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect P wave onset and offset.
- **Step 3**: Compute the time difference.
- **Tools**: `neurokit2` or manual detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify P wave boundaries, calculate duration.
  - Output: P wave durations in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract P wave durations
p_onsets = info['ECG_P_Onsets']
p_offsets = info['ECG_P_Offsets']
p_durations = [(p_offsets[i] - p_onsets[i]) / sampling_rate for i in range(min(len(p_onsets), len(p_offsets)))]
print("P Wave Durations (seconds):", p_durations)
```

**Explanation**: The code detects P wave boundaries and computes their duration. P waves are small, so noise reduction is critical.

---

### 6. T Wave Duration
**Description**: The time from the start to the end of the T wave, showing ventricular repolarization time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect T wave onset and offset.
- **Step 3**: Compute the time difference.
- **Tools**: `neurokit2` or manual detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify T wave boundaries, calculate duration.
  - Output: T wave durations in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract T wave durations
t_onsets = info['ECG_T_Onsets']
t_offsets = info['ECG_T_Offsets']
t_durations = [(t_offsets[i] - t_onsets[i]) / sampling_rate for i in range(min(len(t_onsets), len(t_offsets)))]
print("T Wave Durations (seconds):", t_durations)
```

**Explanation**: The code computes T wave duration. T waves vary in shape, so accurate detection may require parameter tuning in `neurokit2`.

---

### 7. P Wave Amplitude
**Description**: The height of the P wave, indicating atrial depolarization strength.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect P wave peaks.
- **Step 3**: Measure the amplitude (voltage) at the P peak relative to the baseline.
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify P peaks, measure amplitude.
  - Output: P wave amplitudes in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract P wave amplitudes
p_peaks = info['ECG_P_Peaks']
p_amplitudes = ecg_cleaned[p_peaks]
print("P Wave Amplitudes (mV):", p_amplitudes)
```

**Explanation**: The code extracts the signal value at P peaks. Baseline correction is important to ensure accurate amplitude measurements.

---

### 8. R Wave Amplitude
**Description**: The height of the R peak, indicating ventricular depolarization strength.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect R peaks.
- **Step 3**: Measure the amplitude at the R peak relative to the baseline.
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify R peaks, measure amplitude.
  - Output: R wave amplitudes in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract R wave amplitudes
r_peaks = info['ECG_R_Peaks']
r_amplitudes = ecg_cleaned[r_peaks]
print("R Wave Amplitudes (mV):", r_amplitudes)
```

**Explanation**: The code extracts the signal value at R peaks. R peaks are prominent, making them easier to detect than P waves.

---

### 9. Heart Rate Variability (HRV)
**Description**: The variation in RR intervals, reflecting autonomic nervous system activity.
**Extraction Technique**:
- **Step 1**: Extract RR intervals (as above).
- **Step 2**: Compute statistical measures like standard deviation (SDNN), root mean square of successive differences (RMSSD), or pNN50 (percentage of intervals differing by >50 ms).
- **Step 3**: Optionally, analyze HRV over a longer period (e.g., 5 minutes).
- **Tools**: `neurokit2` provides HRV analysis functions.
- **Example**:
  - Input: RR intervals.
  - Process: Compute SDNN, RMSSD, etc.
  - Output: HRV metrics.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract HRV metrics
r_peaks = info['ECG_R_Peaks']
hrv_metrics = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
print("HRV Metrics:", hrv_metrics)
```

**Explanation**: The code uses `nk.hrv_time` to compute HRV metrics like SDNN and RMSSD from R peaks. Longer signals yield more reliable HRV metrics.

---

### 10. ST Segment Elevation/Depression
**Description**: The level of the ST segment (between S wave and T wave), indicating ischemia or infarction.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect S wave offset and T wave onset, identify ST segment.
- **Step 3**: Measure the amplitude of the ST segment relative to the baseline (e.g., isoelectric line at PR segment).
- **Tools**: Manual detection or `neurokit2`.
- **Example**:
  - Input: ECG signal.
  - Process: Identify ST segment, measure amplitude.
  - Output: ST elevation/depression in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract ST segment (approximate as midpoint between S and T)
s_offsets = info['ECG_QRS_Offsets']
t_onsets = info['ECG_T_Onsets']
st_amplitudes = []
for i in range(min(len(s_offsets), len(t_onsets))):
    st_midpoint = (s_offsets[i] + t_onsets[i]) // 2
    st_amplitudes.append(ecg_cleaned[st_midpoint])
print("ST Segment Amplitudes (mV):", st_amplitudes)
```

**Explanation**: The code approximates the ST segment by taking the midpoint between S offset and T onset. In practice, you may need to average over a segment window for accuracy.

---

## 6.2 Frequency-Domain Features: Extraction Techniques

Frequency-domain features analyze the ECG signal’s frequency content, revealing patterns not visible in the time domain. Below are the extraction techniques for the 10 frequency-domain features: **Power Spectral Density (PSD)**, **Low-Frequency (LF) Power**, **High-Frequency (HF) Power**, **LF/HF Ratio**, **Total Power**, **Peak Frequency**, **Spectral Entropy**, **Dominant Frequency**, **Spectral Centroid**, and **Frequency Band Energy**.

### 1. Power Spectral Density (PSD)
**Description**: Shows how signal power is distributed across frequencies.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply Welch’s method or Fast Fourier Transform (FFT) to compute PSD.
- **Step 3**: Output frequency bins and their power values.
- **Tools**: `scipy.signal.welch`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD using Welch’s method.
  - Output: PSD values (mV²/Hz).

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)
print("Frequencies (Hz):", frequencies[:10])
print("PSD Values (mV²/Hz):", psd[:10])
```

**Explanation**: The code uses `signal.welch` to estimate PSD, dividing the signal into overlapping windows for smoother results.

---

### 2. Low-Frequency (LF) Power
**Description**: Power in the 0.04–0.15 Hz band, linked to sympathetic activity.
**Extraction Technique**:
- **Step 1**: Compute PSD (as above).
- **Step 2**: Integrate PSD over the 0.04–0.15 Hz range using numerical integration.
- **Tools**: `scipy.signal.welch`, `numpy.trapz`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, sum power in LF band.
  - Output: LF power in mV².

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate LF power (0.04–0.15 Hz)
lf_band = (0.04, 0.15)
lf_mask = (frequencies >= lf_band[0]) & (frequencies <= lf_band[1])
lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])
print("LF Power (mV²):", lf_power)
```

**Explanation**: The code integrates PSD over the LF band using `np.trapz`. Longer signals (e.g., 5 minutes) are preferred for HRV-related features.

---

### 3. High-Frequency (HF) Power
**Description**: Power in the 0.15–0.4 Hz band, linked to parasympathetic activity.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Integrate PSD over the 0.15–0.4 Hz range.
- **Tools**: `scipy.signal.welch`, `numpy.trapz`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, sum power in HF band.
  - Output: HF power in mV².

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate HF power (0.15–0.4 Hz)
hf_band = (0.15, 0.4)
hf_mask = (frequencies >= hf_band[0]) & (frequencies <= hf_band[1])
hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])
print("HF Power (mV²):", hf_power)
```

**Explanation**: Similar to LF power, but for the HF band. This is useful for assessing respiratory influences on HRV.

---

### 4. LF/HF Ratio
**Description**: The ratio of LF power to HF power, indicating autonomic balance.
**Extraction Technique**:
- **Step 1**: Compute LF and HF power (as above).
- **Step 2**: Divide LF power by HF power.
- **Tools**: `scipy.signal.welch`, `numpy.trapz`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute LF and HF power, calculate ratio.
  - Output: LF/HF ratio (unitless).

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate LF and HF power
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.4)
lf_mask = (frequencies >= lf_band[0]) & (frequencies <= lf_band[1])
hf_mask = (frequencies >= hf_band[0]) & (frequencies <= hf_band[1])
lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])
hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])

# Compute LF/HF ratio
lf_hf_ratio = lf_power / hf_power
print("LF/HF Ratio:", lf_hf_ratio)
```

**Explanation**: The code computes the ratio of LF to HF power, a key HRV metric for autonomic balance.

---

### 5. Total Power
**Description**: The total power across all frequencies, indicating overall signal energy.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Integrate PSD over all frequencies.
- **Tools**: `scipy.signal.welch`, `numpy.trapz`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, sum all power.
  - Output: Total power in mV².

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate total power
total_power = np.trapz(psd, frequencies)
print("Total Power (mV²):", total_power)
```

**Explanation**: The code integrates the entire PSD to get total power, representing the signal’s overall energy.

---

### 6. Peak Frequency
**Description**: The frequency with the highest power in the PSD.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Find the frequency corresponding to the maximum PSD value.
- **Tools**: `scipy.signal.welch`, `numpy.argmax`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, find peak.
  - Output: Peak frequency in Hz.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Find peak frequency
peak_freq = frequencies[np.argmax(psd)]
print("Peak Frequency (Hz):", peak_freq)
```

**Explanation**: The code finds the frequency with the highest PSD value, often related to the heart rate.

---

### 7. Spectral Entropy
**Description**: Measures the randomness or complexity of the frequency spectrum.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Normalize PSD to form a probability distribution.
- **Step 3**: Calculate Shannon entropy of the normalized PSD.
- **Tools**: `scipy.signal.welch`, `scipy.stats.entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, normalize, calculate entropy.
  - Output: Spectral entropy (unitless).

**Code Example**:
```python
import numpy as np
from scipy import signal
from scipy.stats import entropy

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Normalize PSD to form a probability distribution
psd_normalized = psd / np.sum(psd)

# Calculate spectral entropy
spectral_entropy = entropy(psd_normalized)
print("Spectral Entropy:", spectral_entropy)
```

**Explanation**: The code normalizes the PSD and computes its Shannon entropy, indicating frequency spectrum complexity.

---

### 8. Dominant Frequency
**Description**: The most prominent frequency, similar to peak frequency but may consider a broader range.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Identify the frequency with the highest power or average over a dominant band.
- **Tools**: `scipy.signal.welch`, `numpy.argmax`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, find dominant frequency.
  - Output: Dominant frequency in Hz.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Find dominant frequency
dominant_freq = frequencies[np.argmax(psd)]
print("Dominant Frequency (Hz):", dominant_freq)
```

**Explanation**: Similar to peak frequency, this code identifies the frequency with the highest power.

---

### 9. Spectral Centroid
**Description**: The “center of mass” of the frequency spectrum, indicating where power is concentrated.
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Calculate the weighted average of frequencies, using PSD as weights.
- **Tools**: `scipy.signal.welch`, `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, calculate weighted average.
  - Output: Spectral centroid in Hz.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate spectral centroid
spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
print("Spectral Centroid (Hz):", spectral_centroid)
```

**Explanation**: The code computes the weighted average of frequencies, weighted by PSD values.

---

### 10. Frequency Band Energy
**Description**: Energy in a specific frequency band (e.g., 0–10 Hz).
**Extraction Technique**:
- **Step 1**: Compute PSD.
- **Step 2**: Integrate PSD over the desired frequency band.
- **Tools**: `scipy.signal.welch`, `numpy.trapz`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, sum power in band.
  - Output: Band energy in mV².

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 1, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Compute PSD
frequencies, psd = signal.welch(ecg_signal, fs=sampling_rate, nperseg=256)

# Calculate energy in 0–10 Hz band
band = (0, 10)
band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
band_energy = np.trapz(psd[band_mask], frequencies[band_mask])
print("Frequency Band Energy (0–10 Hz, mV²):", band_energy)
```

**Explanation**: The code integrates PSD over a custom frequency band, useful for analyzing specific frequency ranges.

---

## 6.3 Time-Frequency Domain Features: Extraction Techniques

Time-frequency domain features capture how frequencies change over time, ideal for non-stationary ECG signals. Below are the extraction techniques for the 10 time-frequency features: **Short-Time Fourier Transform (STFT)**, **Continuous Wavelet Transform (CWT)**, **Spectrogram**, **Scalogram**, **Wavelet Energy**, **Instantaneous Frequency**, **Time-Frequency Entropy**, **Morlet Wavelet Coefficients**, **Wigner-Ville Distribution**, and **Chirplet Transform**.

### 1. Short-Time Fourier Transform (STFT)
**Description**: Divides the signal into short windows and computes the Fourier Transform for each, showing frequency changes over time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply STFT with a sliding window (e.g., 128 samples).
- **Step 3**: Output a time-frequency matrix.
- **Tools**: `scipy.signal.stft`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute STFT, output time-frequency matrix.
  - Output: STFT coefficients.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute STFT
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)
print("STFT Shape (freqs, times):", Zxx.shape)
```

**Explanation**: The code computes the STFT, producing a matrix where rows are frequencies and columns are time points. The magnitude (`np.abs(Zxx)`) can be used as a feature.

---

### 2. Continuous Wavelet Transform (CWT)
**Description**: Uses wavelets to analyze frequencies at different scales, suitable for non-stationary signals.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply CWT with a wavelet (e.g., Morlet).
- **Step 3**: Output a time-scale matrix.
- **Tools**: `pywt.cwt`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT with Morlet wavelet.
  - Output: CWT coefficients.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute CWT
scales = np.arange(1, 128)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
print("CWT Shape (scales, times):", cwt_matrix.shape)
```

**Explanation**: The code uses `pywt.cwt` with the Morlet wavelet to compute a time-scale matrix, which can be converted to frequencies.

---

### 3. Spectrogram
**Description**: The squared magnitude of the STFT, showing frequency intensity over time.
**Extraction Technique**:
- **Step 1**: Compute STFT.
- **Step 2**: Calculate the squared magnitude of STFT coefficients.
- **Tools**: `scipy.signal.stft`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute STFT, take squared magnitude.
  - Output: Spectrogram matrix.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute spectrogram
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)
spectrogram = np.abs(Zxx)**2
print("Spectrogram Shape (freqs, times):", spectrogram.shape)
```

**Explanation**: The code computes the STFT and squares its magnitude to get the spectrogram, often used as input to CNNs.

---

### 4. Scalogram
**Description**: The squared magnitude of the CWT, showing wavelet coefficient intensity over time and scale.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate the squared magnitude of CWT coefficients.
- **Tools**: `pywt.cwt`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, take squared magnitude.
  - Output: Scalogram matrix.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute scalogram
scales = np.arange(1, 128)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
scalogram = np.abs(cwt_matrix)**2
print("Scalogram Shape (scales, times):", scalogram.shape)
```

**Explanation**: The code computes the CWT and squares its coefficients to get the scalogram, useful for visualizing time-frequency patterns.

---

### 5. Wavelet Energy
**Description**: Energy in specific wavelet scales, indicating strength of frequency bands.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Sum the squared CWT coefficients for specific scales.
- **Tools**: `pywt.cwt`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, sum energy in scales.
  - Output: Wavelet energy values.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute CWT
scales = np.arange(1, 128)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)

# Calculate wavelet energy for each scale
wavelet_energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
print("Wavelet Energy per Scale:", wavelet_energy)
```

**Explanation**: The code sums the squared CWT coefficients across time for each scale, giving energy per scale.

---

### 6. Instantaneous Frequency
**Description**: The frequency at a specific time, derived from time-frequency analysis.
**Extraction Technique**:
- **Step 1**: Compute STFT or CWT.
- **Step 2**: For each time point, find the frequency with the maximum amplitude.
- **Tools**: `scipy.signal.stft`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute STFT, find max frequency per time.
  - Output: Instantaneous frequency values.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute STFT
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)

# Calculate instantaneous frequency
inst_freq = frequencies[np.argmax(np.abs(Zxx), axis=0)]
print("Instantaneous Frequency (Hz):", inst_freq)
```

**Explanation**: The code finds the frequency with the highest amplitude at each time point in the STFT.

---

### 7. Time-Frequency Entropy
**Description**: Measures the complexity of the time-frequency distribution.
**Extraction Technique**:
- **Step 1**: Compute spectrogram or scalogram.
- **Step 2**: Normalize to form a probability distribution.
- **Step 3**: Calculate Shannon entropy.
- **Tools**: `scipy.signal.stft`, `scipy.stats.entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute spectrogram, calculate entropy.
  - Output: Time-frequency entropy.

**Code Example**:
```python
import numpy as np
from scipy import signal
from scipy.stats import entropy

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute spectrogram
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)
spectrogram = np.abs(Zxx)**2

# Normalize spectrogram
spectrogram_normalized = spectrogram / np.sum(spectrogram)

# Calculate time-frequency entropy
tf_entropy = entropy(spectrogram_normalized.flatten())
print("Time-Frequency Entropy:", tf_entropy)
```

**Explanation**: The code normalizes the spectrogram and computes its entropy, indicating time-frequency complexity.

---

### 8. Morlet Wavelet Coefficients
**Description**: Coefficients from CWT using the Morlet wavelet, capturing ECG morphology.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply CWT with Morlet wavelet.
- **Step 3**: Output the coefficient matrix.
- **Tools**: `pywt.cwt`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT with Morlet wavelet.
  - Output: Morlet wavelet coefficients.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute Morlet wavelet coefficients
scales = np.arange(1, 128)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
print("Morlet Wavelet Coefficients Shape (scales, times):", cwt_matrix.shape)
```

**Explanation**: The code computes CWT coefficients using the Morlet wavelet, directly usable as features.

---

### 9. Wigner-Ville Distribution
**Description**: A high-resolution time-frequency method, though sensitive to noise.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Compute the Wigner-Ville distribution (requires custom implementation or libraries).
- **Step 3**: Output the time-frequency matrix.
- **Tools**: Custom code or specialized libraries (e.g., `tfr` in MATLAB).
- **Example**:
  - Input: ECG signal.
  - Process: Compute Wigner-Ville distribution.
  - Output: Wigner-Ville coefficients.
- **Note**: Python libraries for Wigner-Ville are less common, so we’ll simulate a basic approach.

**Code Example**:
```python
import numpy as np
from scipy import signal

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Approximate Wigner-Ville using spectrogram (simplified)
frequencies, times, Zxx = signal.stft(ecg_signal, fs=sampling_rate, nperseg=128)
wigner_ville_approx = np.abs(Zxx)**2  # Simplified placeholder
print("Wigner-Ville Approximation Shape (freqs, times):", wigner_ville_approx.shape)
```

**Explanation**: True Wigner-Ville requires complex computation; this code uses a spectrogram as a placeholder. For accurate Wigner-Ville, consider MATLAB or specialized libraries.

---

### 10. Chirplet Transform
**Description**: Analyzes frequency changes that vary linearly over time, less common in ECG but useful for specific patterns.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply chirplet transform (requires custom implementation or specialized libraries).
- **Step 3**: Output the time-frequency matrix.
- **Tools**: Custom code or research libraries.
- **Example**:
  - Input: ECG signal.
  - Process: Compute chirplet transform (simplified as CWT here).
  - Output: Chirplet coefficients.
- **Note**: Chirplet transform is advanced; we’ll use CWT as a proxy.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Compute CWT as a proxy for chirplet transform
scales = np.arange(1, 128)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
print("Chirplet Transform Approximation Shape (scales, times):", cwt_matrix.shape)
```

**Explanation**: The code uses CWT as a simplified stand-in for the chirplet transform due to limited Python support. For true chirplet analysis, explore research libraries or MATLAB.

---

### Practical Tips for ECG Feature Extraction

1. **Start with Time-Domain Features**: They’re the easiest to compute and understand. Use `neurokit2` to automate peak detection.
2. **Clean Your Signal**: Noise (e.g., baseline wander, muscle artifacts) can ruin feature extraction. Apply filters (e.g., bandpass) before processing.
3. **Use Real Data**: Download ECG data from Physionet (e.g., MIT-BIH Arrhythmia Database) to practice on real signals.
4. **Visualize Results**: Plot your ECG signal, peaks, PSD, or spectrograms to verify your features.
5. **Combine Features**: For machine learning, combine time-domain, frequency-domain, and time-frequency features to capture diverse patterns.
6. **Handle Short Signals**: The example signals are short (1–2 seconds). For HRV or frequency-domain features, use longer signals (e.g., 5 minutes) for better accuracy.
7. **Learn Libraries**: Master `neurokit2`, `biosppy`, `scipy`, and `pywt` for ECG processing. They simplify complex math.
8. **Check Sampling Rate**: Ensure your sampling rate (e.g., 360 Hz) matches your data to avoid errors in time calculations.

### How These Fit into Your PhD

These feature extraction techniques are building blocks for your research in Biomedical Signal Processing. You can:
- Use **time-domain features** (e.g., RR intervals, PR intervals) for simple machine learning models like Random Forests to classify arrhythmias.
- Use **frequency-domain features** (e.g., LF/HF ratio) for heart rate variability analysis, useful for stress or autonomic studies.
- Use **time-frequency features** (e.g., spectrograms, scalograms) as inputs to deep learning models like CNNs for complex pattern detection (e.g., atrial fibrillation).

By practicing these techniques on datasets like MIT-BIH or PTB-XL, you’ll gain hands-on experience for your PhD. Start with the provided code, experiment with real ECG data, and explore how these features improve model performance in detecting heart conditions. If you hit roadblocks, keep tweaking parameters or ask for help—you’re learning a powerful skillset!
