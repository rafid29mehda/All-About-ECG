# Ischemic Heart Disease for ECG Biomedical Signal Processing

## Introduction to Ischemic Heart Disease (IHD) / Coronary Artery Disease (CAD)
**IHD/CAD** is a condition where the heart muscle (myocardium) doesn’t get enough blood flow due to narrowed or blocked coronary arteries, leading to oxygen shortage (ischemia). It’s like a city with clogged roads, reducing power delivery to homes (heart cells). IHD includes **Myocardial Infarction (MI)**, **Unstable Angina**, **Stable Angina**, and **Silent Ischemia**, each with distinct ECG patterns critical for ML/DL analysis. Below, I’ll explain each condition in detail, tailored for your PhD journey in ECG signal processing.

---

## 1. Myocardial Infarction (MI) / Heart Attack

### 1.1 What is Myocardial Infarction?
MI is a severe form of IHD where a coronary artery is completely or nearly completely blocked, causing heart muscle cells to die due to prolonged lack of oxygen. It’s like a power outage in a city block, permanently damaging homes (heart tissue).

### 1.2 Physiology
- **Cause**: Plaque rupture in a coronary artery forms a clot, fully blocking blood flow. Less commonly, vasospasm or embolism.
- **Effect**: Muscle necrosis (infarction) within 20–30 minutes of occlusion. Affects cardiac output and electrical stability, risking arrhythmias.
- **Risk Factors**: Hypertension, smoking, diabetes, high cholesterol, obesity, family history, age (>45 men, >55 women).
- **Complications**: Heart failure, ventricular arrhythmias (e.g., VT/VF), cardiogenic shock, sudden death.

### 1.3 ECG Features
- **Acute Phase (Minutes–Hours)**:
  - **ST Elevation**: >1 mm (0.1 mV) in ≥2 contiguous leads (e.g., II, III, aVF for inferior MI). Convex or “tombstone” shape. Indicates transmural ischemia (STEMI).
  - **Hyperacute T Waves**: Tall, peaked T waves (early sign).
  - **Reciprocal ST Depression**: Opposite leads (e.g., I, aVL for inferior MI).
- **Evolving Phase (Hours–Days)**:
  - **Pathological Q Waves**: >40 ms or >25% of R wave height. Indicate necrosis.
  - **ST Elevation Reduces**: Returns toward baseline.
  - **T Wave Inversion**: Negative T waves in affected leads.
- **Chronic Phase (Weeks–Years)**:
  - **Persistent Q Waves**: Permanent marker of prior MI.
  - **Normalized ST/T**: ST and T waves may normalize or remain abnormal.
- **Lead Patterns**:
  - **Inferior MI**: II, III, aVF (right coronary artery).
  - **Anterior MI**: V1–V4 (left anterior descending).
  - **Lateral MI**: I, aVL, V5–V6 (circumflex).
  - **Posterior MI**: ST depression in V1–V3, tall R waves (mirror of posterior ST elevation).
- **NSTEMI**: Subendocardial infarction, no ST elevation, but ST depression or T wave inversion.

### 1.4 How to Detect
- **Manual Analysis**:
  - Look for ST elevation (>1 mm) in ≥2 contiguous leads, reciprocal changes, evolving Q waves, or T inversion.
  - Confirm with clinical symptoms (chest pain, dyspnea) and troponin levels.
- **Automated Detection**:
  - **Features**: ST segment amplitude, Q wave duration/amplitude, T wave polarity, reciprocal changes.
  - **Algorithms**: Thresholding for ST elevation, wavelet transforms for Q/T waves.
  - **Libraries**: `neurokit2` for ST analysis, `pywavelets` for morphology.
  - **ML/DL**: CNNs for multi-lead ST/Q/T patterns, LSTMs for temporal evolution.
- **Challenges**: Noise mimics ST changes; bundle branch blocks (e.g., LBBB) obscure MI patterns; non-specific changes in NSTEMI.

### 1.5 How to Solve
- **Clinical**:
  - **Emergency (STEMI)**: Percutaneous coronary intervention (PCI) within 90 minutes, thrombolytics if PCI unavailable.
  - **NSTEMI**: Anti-ischemic drugs (nitrates, beta-blockers), anticoagulation (heparin), PCI if high risk.
  - **Long-Term**: Statins, aspirin, beta-blockers, ACE inhibitors, lifestyle changes (diet, exercise, smoking cessation).
- **Signal Processing**:
  - Denoise ECG (band-pass 0.5–40 Hz) to enhance ST/QRS.
  - Extract ST/Q/T features for ML/DL models.
  - Develop real-time MI detection for wearables or ER systems.

### 1.6 Example
Analyze an ECG for MI using Python:
```python
import neurokit2 as nk
import numpy as np
signal = np.random.rand(5000)  # Placeholder ECG (replace with real data)
fs = 500  # Sampling rate
ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
r_peaks = info['ECG_R_Peaks']
# Measure ST deviation (60 ms after J point)
st_values = []
for r in r_peaks:
    j_point = r + int(0.04 * fs)  # J point
    st_point = j_point + int(0.06 * fs)  # 60 ms after J
    if st_point < len(signal):
        baseline = np.mean(signal[r-50:r-10])
        st_values.append(signal[st_point] - baseline)
if np.mean(st_values) > 0.1:  # >1 mm
    print("Possible Myocardial Infarction (STEMI)")
```

**Analogy**: MI is like a city block losing power (blocked artery), causing lights to surge (ST elevation), some to go out (Q waves), and others to flicker (T inversion).

---

## 2. Unstable Angina

### 2.1 What is Unstable Angina?
Unstable angina is chest pain due to partial coronary artery blockage, causing ischemia without infarction. It’s like a city with reduced power supply, flickering lights but no permanent damage.

### 2.2 Physiology
- **Cause**: Plaque rupture or erosion with partial thrombus, reducing blood flow. Vasospasm or demand-supply mismatch (e.g., stress).
- **Effect**: Temporary ischemia, reversible if treated. No muscle death, so no troponin elevation.
- **Risk Factors**: Same as MI (hypertension, smoking, diabetes, etc.).
- **Complications**: Progression to MI, arrhythmias, sudden death.

### 2.3 ECG Features
- **ST Depression**: >1 mm, horizontal or downsloping, in ≥2 contiguous leads. Reflects subendocardial ischemia.
- **T Wave Inversion**: Deep, symmetric, in affected leads (e.g., V1–V4 for anterior ischemia).
- **No ST Elevation**: Unlike STEMI.
- **No Q Waves**: No necrosis.
- **Dynamic Changes**: ECG abnormalities appear during chest pain, may normalize at rest.
- **Leads**: V1–V6 (anterior/lateral), II, III, aVF (inferior).

### 2.4 How to Detect
- **Manual Analysis**:
  - Look for ST depression or T inversion during chest pain, reversible at rest.
  - Rule out MI with normal troponin.
- **Automated Detection**:
  - **Features**: ST depression amplitude, T wave polarity, dynamic changes.
  - **Algorithms**: ST segment analysis, time-series tracking.
  - **Libraries**: `neurokit2` for ST/T, `scipy` for dynamic analysis.
  - **ML/DL**: CNNs for ST/T patterns, RNNs for temporal changes.
- **Challenges**: Subtle ST/T changes, noise interference, non-specific findings.

### 2.5 How to Solve
- **Clinical**:
  - **Acute**: Nitrates (sublingual nitroglycerin), antiplatelets (aspirin, clopidogrel), anticoagulation (heparin), beta-blockers.
  - **Long-Term**: PCI or coronary artery bypass grafting (CABG) if high risk, same as MI (statins, lifestyle).
- **Signal Processing**:
  - Enhance ST/T with band-pass filter (0.5–40 Hz).
  - Track dynamic ST changes for ML/DL models.
  - Develop stress-test ECG analysis systems.

### 2.6 Example
Detect unstable angina:
```python
st_depressions = [v for v in st_values if v < -0.1]  # >1 mm depression
if len(st_depressions) > 0 and np.mean(st_values) < -0.05:
    print("Possible Unstable Angina")
```

**Analogy**: Unstable angina is like dimming lights in a city (ST depression) due to a partial power cut, which stabilizes when demand drops (rest).

---

## 3. Stable Angina

### 3.1 What is Stable Angina?
Stable angina is predictable chest pain triggered by exertion or stress, caused by fixed coronary artery narrowing. It’s like a city with limited power lines, struggling only during peak demand (exercise).

### 3.2 Physiology
- **Cause**: Chronic, stable plaque narrows artery (>50%), limiting blood flow during increased demand (e.g., exercise).
- **Effect**: Reversible ischemia during exertion, relieved by rest or nitrates. No muscle damage.
- **Risk Factors**: Same as MI (hypertension, smoking, etc.).
- **Complications**: Progression to unstable angina or MI if untreated.

### 3.3 ECG Features
- **At Rest**: Normal ECG in most cases.
- **During Stress (e.g., Exercise Test)**:
  - **ST Depression**: >1 mm, horizontal/downsloping, in ≥2 contiguous leads.
  - **T Wave Inversion**: Less common, may occur during peak ischemia.
  - **No ST Elevation or Q Waves**: No infarction.
- **Reversible**: ECG normalizes within minutes of rest.
- **Leads**: V4–V6 (lateral), II, III, aVF (inferior).

### 3.4 How to Detect
- **Manual Analysis**:
  - Perform stress test (treadmill ECG), look for ST depression during exertion.
  - Confirm with clinical history (predictable pain).
- **Automated Detection**:
  - **Features**: ST depression during stress, recovery time.
  - **Algorithms**: ST segment tracking, stress-test analysis.
  - **Libraries**: `neurokit2` for stress ECG, `scipy` for time-series.
  - **ML/DL**: CNNs for stress-induced ST changes.
- **Challenges**: Subtle changes, false positives from non-ischemic causes (e.g., LVH).

### 3.5 How to Solve
- **Clinical**:
  - **Acute**: Sublingual nitroglycerin during episodes.
  - **Long-Term**: Beta-blockers, calcium channel blockers, nitrates, statins, PCI/CABG if severe, lifestyle changes.
- **Signal Processing**:
  - Analyze stress-test ECGs for dynamic ST changes.
  - Develop ML/DL models for automated stress-test interpretation.

### 3.6 Example
Simulate stress-test analysis:
```python
# Assume stress_signal is ECG during exercise
stress_signal = np.random.rand(5000)  # Placeholder
stress_ecg, stress_info = nk.ecg_process(stress_signal, sampling_rate=fs)
stress_st_values = []
for r in stress_info['ECG_R_Peaks']:
    j_point = r + int(0.04 * fs)
    st_point = j_point + int(0.06 * fs)
    if st_point < len(stress_signal):
        baseline = np.mean(stress_signal[r-50:r-10])
        stress_st_values.append(stress_signal[st_point] - baseline)
if np.mean(stress_st_values) < -0.1:
    print("Possible Stable Angina (Stress-Induced ST Depression)")
```

**Analogy**: Stable angina is like a city’s power grid handling normal loads but dimming lights (ST depression) during rush hour (exercise).

---

## 4. Silent Ischemia

### 4.1 What is Silent Ischemia?
Silent ischemia is reduced blood flow to the heart without symptoms, often detected only by ECG or imaging. It’s like a city with power issues that go unnoticed until checked.

### 4.2 Physiology
- **Cause**: Partial coronary artery narrowing or microvascular dysfunction, similar to angina but asymptomatic.
- **Effect**: Reversible ischemia, no infarction, but increases risk of MI or sudden death.
- **Risk Factors**: Diabetes (neuropathy masks pain), elderly, prior MI, same as IHD.
- **Complications**: Undetected progression to MI, heart failure.

### 4.3 ECG Features
- **ST Depression**: >1 mm, horizontal/downsloping, during stress or ambulatory monitoring (Holter).
- **T Wave Inversion**: May occur, less specific.
- **No Symptoms**: Key differentiator from angina.
- **No ST Elevation or Q Waves**: No infarction.
- **Leads**: Similar to angina (V4–V6, II, III, aVF).

### 4.4 How to Detect
- **Manual Analysis**:
  - Use Holter monitoring or stress tests to find asymptomatic ST depression.
  - Confirm with imaging (e.g., stress echo, myocardial perfusion).
- **Automated Detection**:
  - **Features**: ST depression, T wave changes, frequency of episodes.
  - **Algorithms**: Continuous ST monitoring, event detection.
  - **Libraries**: `neurokit2` for ST, `scipy` for event analysis.
  - **ML/DL**: CNNs for silent ST detection in long-term ECGs.
- **Challenges**: Subtle, intermittent changes; requires long-term monitoring.

### 4.5 How to Solve
- **Clinical**:
  - **Treat Asymptomatic IHD**: Beta-blockers, statins, aspirin, PCI/CABG if severe.
  - **Monitor**: Regular stress tests or Holter for high-risk patients (e.g., diabetics).
  - **Lifestyle**: Same as MI (diet, exercise, smoking cessation).
- **Signal Processing**:
  - Analyze long-term ECGs (e.g., Holter) for ST events.
  - Develop ML/DL models for silent ischemia detection in wearables.

### 4.6 Example
Detect silent ischemia in Holter data:
```python
# Assume holter_signal is 24-hour ECG
holter_signal = np.random.rand(5000)  # Placeholder
holter_ecg, holter_info = nk.ecg_process(holter_signal, sampling_rate=fs)
holter_st_values = []
for r in holter_info['ECG_R_Peaks']:
    j_point = r + int(0.04 * fs)
    st_point = j_point + int(0.06 * fs)
    if st_point < len(holter_signal):
        baseline = np.mean(holter_signal[r-50:r-10])
        holter_st_values.append(holter_signal[st_point] - baseline)
if np.any(np.array(holter_st_values) < -0.1):
    print("Possible Silent Ischemia (Asymptomatic ST Depression)")
```

**Analogy**: Silent ischemia is like a city’s hidden power fluctuations (ST depression) that only a meter (ECG) detects, with no complaints from residents.

---

## End-to-End Example: Analyzing ECG for IHD/CAD Conditions

Let’s imagine you’re a PhD student analyzing an ECG from the **PTB-XL Database** to detect IHD conditions (MI, unstable/stable angina, silent ischemia) for an ML/DL project. You’ll preprocess the signal, extract features (e.g., ST segment, Q waves), apply diagnostic rules, train a CNN to classify conditions, and visualize results.

### Step 1: Load Data (WFDB)
```python
import wfdb
import numpy as np

# Load ECG record (10 seconds, 500 Hz)
record = wfdb.rdrecord('ptb-xl/00100_hr', sampto=5000)  # PTB-XL record
signal = record.p_signal[:, 1]  # Lead II
fs = record.fs  # 500 Hz
```

**What’s Happening**: We load a 10-second ECG from PTB-XL, a large dataset with labeled IHD conditions (e.g., MI, ischemia). Lead II is chosen for clear ST and P waves.

**Analogy**: This is like opening a power grid log (ECG) to check for outages (MI) or flickers (ischemia).

### Step 2: Preprocess Signal (SciPy)
```python
from scipy.signal import butter, filtfilt, iirnotch

def preprocess_ecg(data, fs):
    # Band-pass filter (0.5–40 Hz)
    nyq = 0.5 * fs
    b, a = butter(4, [low, high], btype='band')
    data = filtfilt(b, a, data)
    # Notch filter (60 Hz)
    b, a = iirnotch(60, 30, fs)
    return filtfilt(b, a, data)
signal_clean = preprocess_ecg(signal, fs)
```

**What’s Happening**: We filter out baseline wander (<0.5 Hz), muscle noise (>40 Hz), and 60 Hz interference to enhance ST, QRS, and T waves.

**Analogy**: This is like clearing static from a power meter to read voltage (ST segment) accurately.

### Step 3: Fiducial Point Detection (NeuroKit2)
```python
import neurokit2 as nk

# Detect P, QRS, T waves
ecg_signals, info = nk.ecg_process(signal_clean, sampling_rate=fs)
r_peaks = info['ECG_R_Peaks']
q_peaks = info['ECG_Q_Peaks']
s_peaks = info['ECG_S_Peaks']
t_peaks = info['ECG_T_Peaks']
```

**What’s Happening**: `neurokit2` locates R peaks (QRS anchors), Q/S for QRS boundaries, and T for repolarization, enabling ST and Q wave analysis.

**Analogy**: This is like marking key events in a power log—spikes (R peaks), dips (Q waves), and recovery (T waves).

### Step 4: Feature Extraction
```python
# RR intervals and heart rate
rr_intervals = np.diff(r_peaks) / fs
heart_rate = 60 / np.mean(rr_intervals)

# ST deviation and Q wave amplitude
st_values = []
q_amplitudes = []
for r, q, s in zip(r_peaks, q_peaks, s_peaks):
    # ST deviation (60 ms after J point)
    j_point = s + int(0.04 * fs)
    st_point = j_point + int(0.06 * fs)
    if st_point < len(signal_clean):
        baseline = np.mean(signal_clean[r-50:r-10])
        st_values.append(signal_clean[st_point] - baseline)
    # Q wave amplitude
    if not np.isnan(q):
        q_amplitudes.append(np.min(signal_clean[int(q-10):int(q+10)]))
# T wave inversion (simplified)
t_inversions = [signal_clean[t] < 0 for t in t_peaks if not np.isnan(t)]

print(f"Heart Rate: {heart_rate:.1f} BPM, ST Mean: {np.mean(st_values):.3f} mV")
print(f"Q Amplitudes: {np.mean(q_amplitudes):.3f} mV, T Inversions: {sum(t_inversions)}")
```

**What’s Happening**: We extract ST deviation (elevation/depression), Q wave amplitude (pathological in MI), and T wave polarity (inversion in ischemia). These are key IHD features.

**Analogy**: This is like measuring voltage spikes (ST elevation), outages (Q waves), and reversed currents (T inversion) in a power grid.

### Step 5: Diagnostic Rules
```python
diagnoses = []

# Myocardial Infarction
if np.mean(st_values) > 0.1:  # ST elevation >1 mm
    diagnoses.append('Myocardial Infarction (STEMI)')
if np.any(np.array(q_amplitudes) < -0.25 * np.mean(signal_clean[r_peaks])):
    diagnoses.append('Prior MI (Pathological Q Waves)')

# Unstable Angina
if np.mean(st_values) < -0.1 and sum(t_inversions) > len(t_peaks) * 0.5:
    diagnoses.append('Unstable Angina')

# Stable Angina (assume stress test)
stress_st_values = st_values  # Placeholder for stress test
if np.mean(stress_st_values) < -0.1:
    diagnoses.append('Stable Angina (Stress-Induced)')

# Silent Ischemia (assume Holter-like monitoring)
if np.any(np.array(st_values) < -0.1) and heart_rate < 100:
    diagnoses.append('Silent Ischemia')

print(f"Diagnoses: {diagnoses if diagnoses else 'No IHD Detected'}")
```

**What’s Happening**: Rules check ST elevation/Q waves (MI), ST depression/T inversion (unstable angina), stress-induced ST depression (stable angina), and asymptomatic ST depression (silent ischemia). These align with AHA/ESC guidelines.

**Analogy**: This is like a power engineer checking for major outages (MI), flickering during storms (unstable angina), strain during peak use (stable angina), or hidden fluctuations (silent ischemia).

### Step 6: Deep Learning Model (TensorFlow)
We’ll train a CNN to classify ECG segments as **Normal**, **MI**, **Ischemia (Angina/Silent)**, using segmented ECGs around R peaks.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Segment ECG (300 ms window, ~150 samples at 500 Hz)
segments = []
labels = []
window_size = 150
for i, r in enumerate(r_peaks):
    if r >= window_size//2 and r < len(signal_clean) - window_size//2:
        segment = signal_clean[r - window_size//2:r + window_size//2]
        if len(segment) == window_size:
            segments.append(segment)
            # Placeholder labels: simulate PTB-XL labels
            if st_values[i] > 0.1:
                labels.append(1)  # MI
            elif st_values[i] < -0.1:
                labels.append(2)  # Ischemia
            else:
                labels.append(0)  # Normal

# Prepare data
X = np.array(segments)[:, :, np.newaxis]  # Shape: (n_segments, window_size, 1)
y = np.array(labels)
y = tf.keras.utils.to_categorical(y)  # One-hot encode

# Build CNN
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size, 1)),
    Conv1D(64, 5, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Normal, MI, Ischemia
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Predict on first 5 segments
predictions = model.predict(X[:5])
pred_labels = ['Normal', 'MI', 'Ischemia']
print("Predictions:", [pred_labels[np.argmax(p)] for p in predictions])
```

**What’s Happening**: We segment ECGs into 300-ms windows around R peaks, assign labels based on ST features (simulated for demo), and train a CNN to classify segments. The CNN learns ST elevation (MI) and depression (ischemia) patterns. In practice, use PTB-XL labels and balance classes.

**Analogy**: The CNN is like a power engineer learning to spot major outages (MI) or flickering (ischemia) by studying voltage logs (ECG segments).

## Step 7: Visualize Results (Matplotlib)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(signal_clean, label='Filtered ECG', alpha=0.7)
plt.plot(r_peaks, signal_clean[r_peaks], 'ro', label='R Peaks')
plt.plot(t_peaks, signal_clean[t_peaks], 'bo', label='T Peaks')
# Annotate diagnoses
for i, r in enumerate(r_peaks[:5]):  # Limit to 5
    if diagnoses:
        plt.text(r, signal_clean[r] + 0.1, diagnoses[i//2], fontsize=8)
plt.title('ECG with Detected IHD Conditions')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True)
plt.show()
```

**What’s Happening**: The plot shows the cleaned ECG with R peaks (red), T peaks (blue), and text labels for detected IHD conditions, visualizing rule-based and ML/DL results.

## Step 8: Summarize
- **Findings**: We loaded a PTB-XL ECG, preprocessed it, detected fiducial points, extracted ST/Q/T features, applied diagnostic rules to identify MI, angina, and silent ischemia, and trained a CNN for classification. The visualization confirmed detected abnormalities.
- **Outcome**: The pipeline produces features and classifications for ML/DL research, suitable for real-time monitoring or large-scale studies.
- **Next Steps**:
  - Use full PTB-XL dataset with 12-lead ECGs for multi-lead analysis.
  - Balance classes (e.g., oversample ischemia) for better CNN performance.
  - Implement real-time ST detection for wearables.
  - Explore advanced DL (e.g., LSTMs for temporal ST evolution, multi-lead transformers).

---

## Tips for PhD Preparation
- **Practice**: Download PTB-XL or MIT-BIH ECGs from PhysioNet and run this example. Try MI records (e.g., PTB-XL 00101_hr).
- **Visualize**: Plot ECGs with ST elevation/depression to understand patterns.
- **Analogies**: Recall MI as a power outage (ST elevation), unstable angina as flickering (ST depression), stable angina as rush-hour strain (stress ST), and silent ischemia as hidden faults (asymptomatic ST).
- **ML/DL Focus**::
  - Use PTB-XL for IHD classification.
  - Experiment with SVMs for ST-based rules vs. CNNs for raw ECG.
  - Study papers from PhysioNet/CinC for SOTA ECG methods.
- **Tools**: Master `wfdb`, `neurokit2`, `scipy`, `tensorflow`. Explore `pywavelets` for ST denoising.
- **Research Ideas**::
  - Real-time MI detection in wearables.
  - Differentiating STEMI vs. NSTEMI with multi-lead features.
  - Predicting silent ischemia in diabetics using Holter data.
