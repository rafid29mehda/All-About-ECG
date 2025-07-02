## 6.4 Morphological Features: Extraction Techniques

Morphological features describe the shape and size of ECG waves (P, QRS, T). Below are the extraction techniques for the 10 morphological features: **QRS Amplitude**, **T-Wave Shape**, **P Wave Amplitude**, **QRS Width**, **T-Wave Amplitude**, **ST Segment Slope**, **P Wave Shape**, **Q Wave Depth**, **T-Wave Asymmetry**, and **QRS Morphology Variability**.

### 1. QRS Amplitude
**Description**: The height of the R peak in the QRS complex, indicating ventricular depolarization strength.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal to remove noise (e.g., baseline wander).
- **Step 2**: Detect R peaks using a peak detection algorithm (e.g., Pan-Tompkins).
- **Step 3**: Measure the signal amplitude (voltage) at each R peak relative to the baseline.
- **Tools**: `neurokit2` for automated peak detection, `numpy` for calculations.
- **Example**:
  - Input: ECG signal (mV, sampled at 360 Hz).
  - Process: Detect R peaks, extract amplitude values.
  - Output: QRS amplitudes in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal (2 seconds, 360 Hz)
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract QRS amplitude
r_peaks = info['ECG_R_Peaks']
qrs_amplitudes = ecg_cleaned[r_peaks]
print("QRS Amplitudes (mV):", qrs_amplitudes)
```

**Explanation**: The code cleans the ECG signal, detects R peaks using `neurokit2`, and extracts the amplitude at each R peak. For real data, replace the simulated signal with a Physionet dataset (e.g., using `wfdb`).

---

### 2. T-Wave Shape
**Description**: The shape of the T wave (e.g., peaked, flat, inverted), indicating ventricular repolarization characteristics.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect T wave peaks.
- **Step 3**: Analyze T peak amplitude to classify shape:
  - Positive (>0.1 mV): Peaked (normal or exaggerated).
  - Near zero (<0.1 mV): Flat.
  - Negative (<0 mV): Inverted (possible ischemia).
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Detect T peaks, classify based on amplitude.
  - Output: T-wave shape labels (e.g., “Peaked”, “Flat”, “Inverted”).

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract T-wave shape
t_peaks = info['ECG_T_Peaks']
t_amplitudes = ecg_cleaned[t_peaks]
t_shapes = []
for amp in t_amplitudes:
    if abs(amp) < 0.1:
        t_shapes.append("Flat")
    elif amp > 0:
        t_shapes.append("Peaked")
    else:
        t_shapes.append("Inverted")
print("T-Wave Shapes:", t_shapes)
```

**Explanation**: The code detects T peaks and classifies their shape based on amplitude thresholds. For more advanced shape analysis, consider curve fitting or area under the T wave.

---

### 3. P Wave Amplitude
**Description**: The height of the P wave, indicating atrial depolarization strength.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect P wave peaks.
- **Step 3**: Measure the amplitude at P peaks relative to the baseline.
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Detect P peaks, extract amplitude.
  - Output: P wave amplitudes in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract P wave amplitude
p_peaks = info['ECG_P_Peaks']
p_amplitudes = ecg_cleaned[p_peaks]
print("P Wave Amplitudes (mV):", p_amplitudes)
```

**Explanation**: The code extracts the amplitude at P peaks. P waves are small, so ensure the signal is clean to avoid false detections.

---

### 4. QRS Width
**Description**: The duration of the QRS complex, showing ventricular depolarization time.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect QRS complex boundaries (Q onset and S offset).
- **Step 3**: Compute the time difference between Q onset and S offset.
- **Tools**: `neurokit2` for boundary detection.
- **Example**:
  - Input: ECG signal.
  - Process: Identify QRS onset and offset, calculate duration.
  - Output: QRS widths in seconds.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract QRS width
qrs_onsets = info['ECG_QRS_Onsets']
qrs_offsets = info['ECG_QRS_Offsets']
qrs_widths = [(qrs_offsets[i] - qrs_onsets[i]) / sampling_rate for i in range(min(len(qrs_onsets), len(qrs_offsets)))]
print("QRS Widths (seconds):", qrs_widths)
```

**Explanation**: The code computes the time between QRS onset and offset. Accurate boundary detection requires a high-quality signal.

---

### 5. T-Wave Amplitude
**Description**: The height of the T wave, indicating ventricular repolarization strength.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect T wave peaks.
- **Step 3**: Measure the amplitude at T peaks relative to the baseline.
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Detect T peaks, extract amplitude.
  - Output: T-wave amplitudes in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract T-wave amplitude
t_peaks = info['ECG_T_Peaks']
t_amplitudes = ecg_cleaned[t_peaks]
print("T-Wave Amplitudes (mV):", t_amplitudes)
```

**Explanation**: The code extracts the amplitude at T peaks. T waves vary in shape, so ensure proper baseline correction.

---

### 6. ST Segment Slope
**Description**: The slope of the ST segment (between S wave and T wave), indicating ischemia or injury.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect S wave offset and T wave onset to define the ST segment.
- **Step 3**: Fit a line to the ST segment and calculate its slope (change in amplitude per sample).
- **Tools**: `neurokit2` for boundaries, `numpy` for slope calculation.
- **Example**:
  - Input: ECG signal.
  - Process: Identify ST segment, compute slope.
  - Output: ST segment slopes (mV/sample).

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract ST segment slope
s_offsets = info['ECG_QRS_Offsets']
t_onsets = info['ECG_T_Onsets']
st_slopes = []
for i in range(min(len(s_offsets), len(t_onsets))):
    st_segment = ecg_cleaned[s_offsets[i]:t_onsets[i]]
    if len(st_segment) > 1:
        x = np.arange(len(st_segment))
        slope = np.polyfit(x, st_segment, 1)[0]  # Linear fit slope
        st_slopes.append(slope)
print("ST Segment Slopes (mV/sample):", st_slopes)
```

**Explanation**: The code extracts the ST segment and fits a linear line to compute the slope. Short segments may lead to noisy slopes, so consider averaging.

---

### 7. P Wave Shape
**Description**: The shape of the P wave (e.g., notched, broad), indicating atrial abnormalities.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect P wave onset and offset to isolate the P wave.
- **Step 3**: Analyze the P wave for characteristics like notching (multiple peaks) or breadth (duration).
- **Tools**: `neurokit2` for boundaries.
- **Example**:
  - Input: ECG signal.
  - Process: Isolate P wave, check for notching or duration.
  - Output: P wave shape labels (e.g., “Normal”, “Notched”).

**Code Example**:
```python
import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract P wave shape
p_onsets = info['ECG_P_Onsets']
p_offsets = info['ECG_P_Offsets']
p_shapes = []
for i in range(min(len(p_onsets), len(p_offsets))):
    p_wave = ecg_cleaned[p_onsets[i]:p_offsets[i]]
    peaks, _ = find_peaks(p_wave, height=0.05)
    if len(peaks) > 1:
        p_shapes.append("Notched")
    else:
        p_shapes.append("Normal")
print("P Wave Shapes:", p_shapes)
```

**Explanation**: The code isolates the P wave and checks for multiple peaks to detect notching. Adjust the height threshold based on the signal.

---

### 8. Q Wave Depth
**Description**: The depth (negative amplitude) of the Q wave, indicating possible past infarction.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect Q wave peaks (local minima before R peaks).
- **Step 3**: Measure the amplitude at Q peaks relative to the baseline.
- **Tools**: `neurokit2` for peak detection.
- **Example**:
  - Input: ECG signal.
  - Process: Detect Q peaks, extract amplitude.
  - Output: Q wave depths in millivolts.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract Q wave depth
q_peaks = info['ECG_Q_Peaks']
q_depths = ecg_cleaned[q_peaks]
print("Q Wave Depths (mV):", q_depths)
```

**Explanation**: The code extracts the amplitude at Q peaks. Q waves are small, so high signal quality is crucial.

---

### 9. T-Wave Asymmetry
**Description**: The asymmetry of the T wave, indicating repolarization abnormalities.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Detect T wave onset, peak, and offset.
- **Step 3**: Compare the areas or slopes of the ascending and descending parts of the T wave.
- **Tools**: `neurokit2` for boundaries, `numpy` for calculations.
- **Example**:
  - Input: ECG signal.
  - Process: Isolate T wave, compute asymmetry.
  - Output: T-wave asymmetry ratios.

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract T-wave asymmetry
t_onsets = info['ECG_T_Onsets']
t_peaks = info['ECG_T_Peaks']
t_offsets = info['ECG_T_Offsets']
t_asymmetries = []
for i in range(min(len(t_onsets), len(t_peaks), len(t_offsets))):
    t_wave_ascending = ecg_cleaned[t_onsets[i]:t_peaks[i]]
    t_wave_descending = ecg_cleaned[t_peaks[i]:t_offsets[i]]
    area_ascending = np.trapz(t_wave_ascending)
    area_descending = np.trapz(t_wave_descending)
    asymmetry = abs(area_ascending / area_descending) if area_descending != 0 else np.inf
    t_asymmetries.append(asymmetry)
print("T-Wave Asymmetry Ratios:", t_asymmetries)
```

**Explanation**: The code computes the ratio of areas under the ascending and descending T wave parts. A ratio far from 1 indicates asymmetry.

---

### 10. QRS Morphology Variability
**Description**: The variation in QRS complex shapes across beats, indicating irregular heart activity.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Segment QRS complexes using R peaks.
- **Step 3**: Compute a similarity metric (e.g., correlation) between QRS complexes.
- **Tools**: `neurokit2` for segmentation, `numpy` for correlation.
- **Example**:
  - Input: ECG signal.
  - Process: Segment QRS complexes, compute correlation variability.
  - Output: QRS morphology variability (e.g., standard deviation of correlations).

**Code Example**:
```python
import neurokit2 as nk
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Clean and process ECG
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# Extract QRS morphology variability
r_peaks = info['ECG_R_Peaks']
qrs_segments = []
window = int(0.1 * sampling_rate)  # 100 ms window around R peak
for r in r_peaks:
    start = max(0, r - window)
    end = min(len(ecg_cleaned), r + window)
    qrs_segments.append(ecg_cleaned[start:end])
qrs_segments = [seg for seg in qrs_segments if len(seg) == len(qrs_segments[0])]  # Ensure same length
correlations = [np.corrcoef(qrs_segments[i], qrs_segments[0])[0, 1] for i in range(1, len(qrs_segments))]
variability = np.std(correlations) if correlations else 0
print("QRS Morphology Variability (Std of Correlations):", variability)
```

**Explanation**: The code segments QRS complexes and computes the standard deviation of their correlations to quantify shape variability.

---

## 6.5 Statistical Features: Extraction Techniques

Statistical features summarize the ECG signal’s distribution. Below are the extraction techniques for **Mean**, **Variance**, **Skewness**, **Kurtosis**, **Standard Deviation**, **Median**, **Range**, **Interquartile Range (IQR)**, **Root Mean Square (RMS)**, and **Coefficient of Variation**.

### 1. Mean
**Description**: The average amplitude of the ECG signal.
**Extraction Technique**:
- **Step 1**: Select an ECG segment (e.g., full signal or per heartbeat).
- **Step 2**: Compute the arithmetic mean of the signal values.
- **Tools**: `numpy.mean`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate mean.
  - Output: Mean in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract mean
mean = np.mean(ecg_signal)
print("Mean (mV):", mean)
```

**Explanation**: The code computes the average amplitude of the ECG signal.

---

### 2. Variance
**Description**: How much the signal’s amplitude varies from the mean.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute the variance (average of squared deviations from the mean).
- **Tools**: `numpy.var`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate variance.
  - Output: Variance in mV².

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract variance
variance = np.var(ecg_signal)
print("Variance (mV²):", variance)
```

**Explanation**: The code computes the variance, indicating signal spread.

---

### 3. Skewness
**Description**: Measures the asymmetry of the signal’s amplitude distribution.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute skewness using the third standardized moment.
- **Tools**: `scipy.stats.skew`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate skewness.
  - Output: Skewness (unitless).

**Code Example**:
```python
import numpy as np
from scipy import stats

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract skewness
skewness = stats.skew(ecg_signal)
print("Skewness:", skewness)
```

**Explanation**: The code computes skewness, where positive values indicate a right-skewed distribution.

---

### 4. Kurtosis
**Description**: Measures the “peakedness” or presence of outliers in the signal’s distribution.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute kurtosis using the fourth standardized moment.
- **Tools**: `scipy.stats.kurtosis`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate kurtosis.
  - Output: Kurtosis (unitless).

**Code Example**:
```python
import numpy as np
from scipy import stats

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract kurtosis
kurtosis = stats.kurtosis(ecg_signal)
print("Kurtosis:", kurtosis)
```

**Explanation**: The code computes kurtosis, where high values indicate sharp peaks or outliers.

---

### 5. Standard Deviation
**Description**: The square root of variance, showing typical deviation from the mean.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute the standard deviation.
- **Tools**: `numpy.std`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate standard deviation.
  - Output: Standard deviation in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract standard deviation
std = np.std(ecg_signal)
print("Standard Deviation (mV):", std)
```

**Explanation**: The code computes the standard deviation, a common measure of variability.

---

### 6. Median
**Description**: The middle value of the signal’s amplitudes.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute the median.
- **Tools**: `numpy.median`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate median.
  - Output: Median in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract median
median = np.median(ecg_signal)
print("Median (mV):", median)
```

**Explanation**: The code computes the median, robust to outliers compared to the mean.

---

### 7. Range
**Description**: The difference between the maximum and minimum signal values.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute the range (max - min).
- **Tools**: `numpy.ptp`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate range.
  - Output: Range in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract range
range_val = np.ptp(ecg_signal)
print("Range (mV):", range_val)
```

**Explanation**: The code uses `np.ptp` (peak-to-peak) to compute the range.

---

### 8. Interquartile Range (IQR)
**Description**: The range between the 25th and 75th percentiles, showing the middle 50% of values.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute the 25th and 75th percentiles, then subtract.
- **Tools**: `numpy.percentile`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate IQR.
  - Output: IQR in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract IQR
iqr = np.percentile(ecg_signal, 75) - np.percentile(ecg_signal, 25)
print("IQR (mV):", iqr)
```

**Explanation**: The code computes the IQR, robust to extreme values.

---

### 9. Root Mean Square (RMS)
**Description**: The square root of the mean of squared values, indicating signal energy.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute RMS (square root of mean of squared amplitudes).
- **Tools**: `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate RMS.
  - Output: RMS in millivolts.

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract RMS
rms = np.sqrt(np.mean(ecg_signal**2))
print("RMS (mV):", rms)
```

**Explanation**: The code computes the RMS, reflecting the signal’s energy.

---

### 10. Coefficient of Variation
**Description**: The standard deviation divided by the mean, showing relative variability.
**Extraction Technique**:
- **Step 1**: Select an ECG segment.
- **Step 2**: Compute standard deviation and mean, then divide.
- **Tools**: `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Calculate coefficient of variation.
  - Output: Coefficient of variation (unitless).

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract coefficient of variation
std = np.std(ecg_signal)
mean = np.mean(ecg_signal)
cov = std / mean if mean != 0 else np.inf
print("Coefficient of Variation:", cov)
```

**Explanation**: The code computes the coefficient of variation, useful for comparing variability across signals.

---

## 6.6 Wavelet-Based Feature Extraction: Extraction Techniques

Wavelet-based features use wavelets to analyze the ECG signal at different scales. Below are the extraction techniques for **Wavelet Coefficients**, **Wavelet Energy**, **Wavelet Entropy**, **Detail Coefficients**, **Approximation Coefficients**, **Wavelet Power Spectrum**, **Scale-Specific Energy Ratio**, **Wavelet Variance**, **Wavelet Skewness**, and **Wavelet Kurtosis**.

### 1. Wavelet Coefficients
**Description**: Raw values from the wavelet transform, showing signal details at different scales.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply Continuous Wavelet Transform (CWT) with a wavelet (e.g., Morlet).
- **Step 3**: Output the coefficient matrix.
- **Tools**: `pywt.cwt`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT.
  - Output: Wavelet coefficient matrix.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet coefficients
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
print("Wavelet Coefficients Shape (scales, times):", cwt_matrix.shape)
```

**Explanation**: The code computes the CWT, producing a matrix of coefficients for each scale and time point.

---

### 2. Wavelet Energy
**Description**: The energy (squared magnitude) of wavelet coefficients at specific scales.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Sum the squared magnitudes of coefficients for each scale.
- **Tools**: `pywt.cwt`, `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, sum squared coefficients per scale.
  - Output: Wavelet energy per scale.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet energy
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
wavelet_energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
print("Wavelet Energy per Scale:", wavelet_energy)
```

**Explanation**: The code sums the squared CWT coefficients along the time axis for each scale.

---

### 3. Wavelet Entropy
**Description**: Measures the randomness of wavelet coefficients.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Normalize squared coefficients to form a probability distribution.
- **Step 3**: Calculate Shannon entropy.
- **Tools**: `pywt.cwt`, `scipy.stats.entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate entropy.
  - Output: Wavelet entropy (unitless).

**Code Example**:
```python
import numpy as np
import pywt
from scipy.stats import entropy

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet entropy
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
cwt_normalized = np.abs(cwt_matrix)**2 / np.sum(np.abs(cwt_matrix)**2)
wavelet_entropy = entropy(cwt_normalized.flatten())
print("Wavelet Entropy:", wavelet_entropy)
```

**Explanation**: The code normalizes the squared CWT coefficients and computes their entropy.

---

### 4. Detail Coefficients (High-Frequency)
**Description**: Coefficients capturing fast changes (e.g., QRS complex) from Discrete Wavelet Transform (DWT).
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply DWT to decompose into detail coefficients.
- **Step 3**: Extract high-frequency coefficients (e.g., level 1–3).
- **Tools**: `pywt.wavedec`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute DWT, extract detail coefficients.
  - Output: Detail coefficients.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract detail coefficients (DWT)
coeffs = pywt.wavedec(ecg_signal, 'db4', level=3)
detail_coeffs = coeffs[1:4]  # Levels 1–3 (high-frequency)
print("Detail Coefficients Shapes:", [len(c) for c in detail_coeffs])
```

**Explanation**: The code uses DWT with the Daubechies wavelet (‘db4’) to extract high-frequency detail coefficients.

---

### 5. Approximation Coefficients (Low-Frequency)
**Description**: Coefficients capturing slow changes (e.g., P, T waves) from DWT.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply DWT to decompose into approximation coefficients.
- **Step 3**: Extract low-frequency coefficients (e.g., level 3 approximation).
- **Tools**: `pywt.wavedec`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute DWT, extract approximation coefficients.
  - Output: Approximation coefficients.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract approximation coefficients (DWT)
coeffs = pywt.wavedec(ecg_signal, 'db4', level=3)
approx_coeffs = coeffs[0]  # Level 3 approximation
print("Approximation Coefficients Shape:", len(approx_coeffs))
```

**Explanation**: The code extracts low-frequency approximation coefficients using DWT.

---

### 6. Wavelet Power Spectrum
**Description**: The distribution of energy across wavelet scales.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate the squared magnitude of coefficients for each scale.
- **Tools**: `pywt.cwt`, `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate power spectrum.
  - Output: Power per scale.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet power spectrum
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
power_spectrum = np.mean(np.abs(cwt_matrix)**2, axis=1)
print("Wavelet Power Spectrum:", power_spectrum)
```

**Explanation**: The code computes the average squared magnitude of CWT coefficients per scale.

---

### 7. Scale-Specific Energy Ratio
**Description**: The proportion of energy in specific scales.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate energy per scale and divide by total energy.
- **Tools**: `pywt.cwt`, `numpy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate energy ratios.
  - Output: Energy ratios per scale.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract scale-specific energy ratio
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
total_energy = np.sum(energy)
energy_ratios = energy / total_energy
print("Scale-Specific Energy Ratios:", energy_ratios)
```

**Explanation**: The code computes the energy per scale and normalizes by total energy.

---

### 8. Wavelet Variance
**Description**: The variance of wavelet coefficients per scale.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate variance of coefficients for each scale.
- **Tools**: `pywt.cwt`, `numpy.var`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate variance per scale.
  - Output: Wavelet variance per scale.

**Code Example**:
```python
import numpy as np
import pywt

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet variance
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
wavelet_variance = np.var(cwt_matrix, axis=1)
print("Wavelet Variance per Scale:", wavelet_variance)
```

**Explanation**: The code computes the variance of CWT coefficients for each scale.

---

### 9. Wavelet Skewness
**Description**: The skewness of wavelet coefficients per scale.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate skewness of coefficients for each scale.
- **Tools**: `pywt.cwt`, `scipy.stats.skew`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate skewness per scale.
  - Output: Wavelet skewness per scale.

**Code Example**:
```python
import numpy as np
import pywt
from scipy.stats import skew

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet skewness
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
wavelet_skewness = skew(cwt_matrix, axis=1)
print("Wavelet Skewness per Scale:", wavelet_skewness)
```

**Explanation**: The code computes the skewness of CWT coefficients for each scale.

---

### 10. Wavelet Kurtosis
**Description**: The kurtosis of wavelet coefficients per scale.
**Extraction Technique**:
- **Step 1**: Compute CWT.
- **Step 2**: Calculate kurtosis of coefficients for each scale.
- **Tools**: `pywt.cwt`, `scipy.stats.kurtosis`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, calculate kurtosis per scale.
  - Output: Wavelet kurtosis per scale.

**Code Example**:
```python
import numpy as np
import pywt
from scipy.stats import kurtosis

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 5 * time) + 0.3 * np.random.randn(len(time))

# Extract wavelet kurtosis
scales = np.arange(1, 64)
cwt_matrix, frequencies = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
wavelet_kurtosis = kurtosis(cwt_matrix, axis=1)
print("Wavelet Kurtosis per Scale:", wavelet_kurtosis)
```

**Explanation**: The code computes the kurtosis of CWT coefficients for each scale.

---

### Practical Tips for ECG Feature Extraction

1. **Use Real Data**: Replace simulated signals with Physionet datasets (e.g., MIT-BIH) using `wfdb`:
   ```python
   import wfdb
   record = wfdb.rdrecord('mitdb/100', sampto=720)
   ecg_signal = record.p_signal[:, 0]
   ```
2. **Clean Signals**: Apply bandpass filters (e.g., 0.5–40 Hz) to remove noise before feature extraction.
3. **Segment Signals**: For morphological and statistical features, segment the ECG around R peaks to analyze individual heartbeats.
4. **Choose Wavelets**: Use Morlet for CWT (good for ECG) or Daubechies for DWT. Experiment with scales to capture relevant frequencies.
5. **Visualize Results**: Plot ECG signals, peaks, or scalograms to verify feature extraction.
6. **Combine Features**: Use these features together in machine learning models (e.g., Random Forest, CNN) for better classification of heart conditions.
7. **Handle Noise**: Noise can distort morphological and wavelet features. Use `neurokit2` or `scipy` filters to preprocess.
8. **Optimize for Research**: For the PhD, test these features on specific conditions (e.g., arrhythmias) and evaluate their impact on model performance.

These techniques provide a solid foundation for extracting ECG features, enabling you to analyze heart signals effectively in the Biomedical Signal Processing research. Practice with these codes, try real datasets, and explore how these features enhance the machine learning models!
