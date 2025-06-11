# Conduction Abnormalities for ECG Biomedical Signal Processing

## Introduction to Conduction Abnormalities
**Conduction abnormalities** occur when the heart’s electrical signals are delayed, blocked, or rerouted, disrupting the normal sequence of atrial and ventricular activation. It’s like a traffic jam or detour in the heart’s electrical highway, affecting how the heartbeat is coordinated. These conditions—**Heart Blocks**, **Bundle Branch Blocks**, and **WPW Syndrome**—are key for ECG analysis in ML/DL, as they produce distinct patterns critical for diagnosis. Below, I’ll break down each condition in a beginner-friendly way, tailored for your PhD journey.

---

## 1. Heart Block (First, Second, and Third Degree)

### 1.1 What is Heart Block?
Heart block is a delay or interruption in the electrical signal traveling from the atria to the ventricles through the atrioventricular (AV) node or His-Purkinje system. It’s like a slow, congested, or broken road between the heart’s upper and lower chambers.

#### First-Degree Heart Block
- **What**: A delay in signal conduction through the AV node, but every atrial impulse reaches the ventricles.
- **Analogy**: A traffic light that takes too long to turn green, slowing cars (signals) but letting all through.

#### Second-Degree Heart Block
- **What**: Some atrial impulses fail to reach the ventricles, causing “dropped” beats.
- **Types**:
  - **Type I (Wenckebach)**: Progressive delay until a beat is dropped.
  - **Type II**: Sudden dropped beats without prior delay.
- **Analogy**: A toll booth that occasionally refuses cars, either after slowing (Type I) or randomly (Type II).

#### Third-Degree Heart Block
- **What**: Complete block; no atrial impulses reach the ventricles, which beat independently via an escape rhythm.
- **Analogy**: A collapsed bridge, forcing the ventricles to find their own slow, backup route.

### 1.2 Physiology
- **Cause**:
  - **First-Degree**: Increased vagal tone (e.g., athletes), medications (beta-blockers), fibrosis, ischemia.
  - **Second-Degree Type I**: Vagal tone, inferior MI, drugs (digoxin).
  - **Second-Degree Type II**: His-Purkinje damage, anterior MI, fibrosis.
  - **Third-Degree**: Severe AV node/His-Purkinje damage, MI, Lyme disease, congenital.
- **Effect**:
  - **First-Degree**: Usually asymptomatic, minor delay.
  - **Second-Degree**: Palpitations, dizziness, or syncope due to dropped beats.
  - **Third-Degree**: Severe bradycardia (30–40 bpm), syncope, heart failure, risk of asystole.
- **Risk Factors**: Aging, heart disease (MI, cardiomyopathy), medications, electrolyte imbalances, infections (e.g., Lyme).
- **Complications**: Progression to higher-degree block, heart failure, sudden cardiac death (third-degree).

### 1.3 ECG Features
- **First-Degree**:
  - **PR Interval**: Prolonged (>200 ms, 0.2 s).
  - **P and QRS**: Normal, every P followed by QRS.
  - **Rhythm**: Regular.
- **Second-Degree Type I (Wenckebach)**:
  - **PR Interval**: Progressively lengthens until a QRS is dropped (no QRS after P).
  - **P Waves**: Regular, more P waves than QRS.
  - **Rhythm**: Irregular QRS due to dropped beats.
- **Second-Degree Type II**:
  - **PR Interval**: Constant, but some P waves not followed by QRS.
  - **P Waves**: Regular, more P waves than QRS.
  - **Rhythm**: Irregular QRS, often in patterns (e.g., 2:1 block).
- **Third-Degree**:
  - **P and QRS Dissociation**: No relationship between P waves and QRS complexes.
  - **P Waves**: Regular, faster than QRS (atrial rate 60–100 bpm).
  - **QRS**: Regular, slow (escape rhythm, 30–40 bpm, wide if ventricular).
  - **Rhythm**: Atria and ventricles independent.
- **Leads**: Lead II for PR clarity, V1 for QRS morphology.

### 1.4 How to Detect
- **Manual Analysis**:
  - **First-Degree**: Measure PR interval (>200 ms).
  - **Second-Degree Type I**: Look for lengthening PR, dropped QRS.
  - **Second-Degree Type II**: Constant PR, occasional dropped QRS.
  - **Third-Degree**: Check for P-QRS dissociation, slow QRS rate.
- **Automated Detection**:
  - **Features**: PR interval, P-to-QRS ratio, QRS rate, P-QRS correlation.
  - **Algorithms**: P and QRS detection, interval analysis.
  - **Libraries**: `neurokit2` for PR/QRS, `scipy` for pattern analysis.
  - **ML/DL**: CNNs for PR/QRS patterns, LSTMs for sequence analysis.
- **Challenges**: Noise obscures P waves; irregular rhythms complicate PR measurement.

### 1.5 How to Solve
- **Clinical**:
  - **First-Degree**: Often benign, monitor, adjust medications if symptomatic.
  - **Second-Degree Type I**: Treat cause (e.g., stop drugs), monitor, pacemaker if severe.
  - **Second-Degree Type II**: Pacemaker often required, treat underlying disease.
  - **Third-Degree**: Emergency pacemaker, atropine (temporary), treat cause (e.g., MI).
- **Signal Processing**:
  - Enhance P/QRS with band-pass filter (0.5–40 Hz).
  - Extract PR intervals and P-QRS patterns for ML/DL models.
  - Develop real-time block detection for pacemakers or monitors.

### 1.6 Example
Detect heart block:
```python
import neurokit2 as nk
import numpy as np
signal = np.random.rand(5000)  # Placeholder ECG
fs = 500
ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
pr_intervals = [(q - p) / fs for p, q in zip(info['ECG_P_Peaks'], info['ECG_Q_Peaks']) if not (np.isnan(p) or np.isnan(q))]
if np.mean(pr_intervals) > 0.2:
    print("First-Degree Heart Block")
elif len(info['ECG_P_Peaks']) > len(info['ECG_R_Peaks']):
    print("Possible Second-Degree Heart Block")
```

**Analogy**: Heart block is like a slow (first-degree), congested (second-degree), or collapsed (third-degree) road between atria and ventricles, delaying or stopping traffic (signals).

---

## 2. Bundle Branch Blocks (Right and Left)

### 2.1 What are Bundle Branch Blocks?
Bundle branch blocks (BBB) occur when the electrical signal is delayed or blocked in the right or left bundle branch, slowing ventricular activation. It’s like a lane closure on a highway, forcing traffic (signals) to take a slower route.

#### Right Bundle Branch Block (RBBB)
- **What**: Block in the right bundle, delaying right ventricular activation.
- **Analogy**: Closing the right lane, slowing traffic to the right side of the city (right ventricle).

#### Left Bundle Branch Block (LBBB)
- **What**: Block in the left bundle, delaying left ventricular activation.
- **Analogy**: Closing the left lane, slowing traffic to the left side of the city (left ventricle).

### 2.2 Physiology
- **Cause**:
  - **RBBB**: Pulmonary hypertension, right ventricular hypertrophy, MI, congenital, fibrosis.
  - **LBBB**: Hypertension, MI, cardiomyopathy, aortic stenosis, fibrosis.
- **Effect**:
  - **RBBB**: Delayed right ventricular contraction, usually benign in healthy individuals.
  - **LBBB**: Delayed left ventricular contraction, often indicates heart disease, affects cardiac efficiency.
- **Risk Factors**: Heart disease, aging, hypertension, MI, congenital defects.
- **Complications**:
  - **RBBB**: Rarely symptomatic, may mask MI on ECG.
  - **LBBB**: Heart failure, complicates MI diagnosis, risk of progression to heart block.

### 2.3 ECG Features
- **RBBB**:
  - **QRS Duration**: Prolonged (>120 ms, 0.12 s).
  - **QRS Morphology**: RSR’ pattern (“rabbit ears”) in V1–V2, wide S wave in I, V5–V6.
  - **ST/T Waves**: Discordant (opposite QRS direction, e.g., ST depression in V1).
  - **P and PR**: Normal unless associated block.
- **LBBB**:
  - **QRS Duration**: Prolonged (>120 ms).
  - **QRS Morphology**: Broad, notched R waves in I, aVL, V5–V6; deep S waves in V1–V3.
  - **ST/T Waves**: Discordant (e.g., ST elevation in leads with negative QRS).
  - **P and PR**: Normal unless associated block.
- **Leads**: V1–V2 (RBBB), V5–V6 (LBBB) for clear morphology.

### 2.4 How to Detect
- **Manual Analysis**:
  - **RBBB**: Look for wide QRS, RSR’ in V1, wide S in I/V6.
  - **LBBB**: Wide QRS, notched R in I/V6, deep S in V1.
- **Automated Detection**:
  - **Features**: QRS duration, RSR’ pattern, notched R, S wave depth.
  - **Algorithms**: QRS morphology analysis, peak detection.
  - **Libraries**: `neurokit2` for QRS, `scipy` for morphology.
  - **ML/DL**: CNNs for QRS pattern recognition.
- **Challenges**: Noise distorts QRS; LBBB obscures MI patterns.

### 2.5 How to Solve
- **Clinical**:
  - **RBBB**: Often benign, treat underlying cause (e.g., pulmonary disease).
  - **LBBB**: Treat heart disease (e.g., hypertension, MI), consider cardiac resynchronization therapy (CRT) for heart failure.
  - **Both**: Monitor, pacemaker if progressing to heart block.
- **Signal Processing**:
  - Enhance QRS with band-pass filter (5–40 Hz).
  - Extract QRS morphology for ML/DL classification.
  - Develop algorithms to differentiate BBB from MI.

### 2.6 Example
Detect BBB:
```python
qrs_widths = [(s - q) / fs for q, s in zip(info['ECG_Q_Peaks'], info['ECG_S_Peaks']) if not (np.isnan(q) or np.isnan(s))]
if np.mean(qrs_widths) > 0.12:
    print("Possible Bundle Branch Block")
```

**Analogy**: BBB is like a lane closure on the right (RBBB) or left (LBBB) highway, slowing traffic (QRS widening) to one side of the heart.

---

## 3. Wolff-Parkinson-White Syndrome (WPW)

### 3.1 What is WPW Syndrome?
WPW is a condition where an accessory pathway (bypass tract) allows electrical signals to travel between atria and ventricles outside the AV node, causing premature ventricular activation. It’s like a secret shortcut bypassing the main road, speeding up traffic unpredictably.

### 3.2 Physiology
- **Cause**: Congenital accessory pathway (e.g., Bundle of Kent), present at birth.
- **Effect**: Pre-excitation of ventricles, leading to fast heart rates (tachycardia) or arrhythmias (e.g., SVT, AF). May be asymptomatic until triggered.
- **Risk Factors**: Congenital, family history, rare acquired causes (e.g., cardiomyopathy).
- **Complications**: Rapid AF or VT, sudden cardiac death (rare, especially if AF conducts rapidly via pathway).

### 3.3 ECG Features
- **Short PR Interval**: <120 ms (0.12 s), due to fast conduction via accessory pathway.
- **Delta Wave**: Slurred upstroke at QRS start, reflecting pre-excitation.
- **Wide QRS**: >120 ms, due to combined pathway and normal conduction.
- **Tachycardia Episodes**: May show SVT (narrow QRS) or AF (irregular, wide QRS if pathway conducts).
- **Leads**: Delta wave visible in multiple leads (e.g., V1–V6, II).

### 3.4 How to Detect
- **Manual Analysis**:
  - Look for short PR, delta wave, wide QRS in resting ECG.
  - Check for tachycardia episodes (SVT, AF).
- **Automated Detection**:
  - **Features**: PR interval, delta wave slope, QRS width.
  - **Algorithms**: QRS onset detection, slope analysis.
  - **Libraries**: `neurokit2` for PR/QRS, `scipy` for delta wave.
  - **ML/DL**: CNNs for delta wave patterns, LSTMs for tachycardia.
- **Challenges**: Delta wave may be subtle; tachycardia mimics other arrhythmias.

### 3.5 How to Solve
- **Clinical**:
  - **Asymptomatic**: Monitor, avoid triggers (e.g., caffeine).
  - **Symptomatic**: Antiarrhythmics (e.g., flecainide), catheter ablation (preferred, >90% success).
  - **Emergency**: Cardioversion for rapid AF, vagal maneuvers for SVT.
- **Signal Processing**:
  - Enhance QRS onset with high-pass filter (5 Hz).
  - Extract delta wave features for ML/DL models.
  - Develop real-time WPW detection for arrhythmia monitors.

### 3.6 Example
Detect WPW:
```python
pr_intervals = [(q - p) / fs for p, q in zip(info['ECG_P_Peaks'], info['ECG_Q_Peaks']) if not (np.isnan(p) or np.isnan(q))]
if np.mean(pr_intervals) < 0.12 and np.mean(qrs_widths) > 0.12:
    print("Possible WPW Syndrome")
```

**Analogy**: WPW is like a shortcut road (accessory pathway) letting cars (signals) reach the ventricles too early, causing traffic jams (delta wave, wide QRS).

---

## End-to-End Example: Analyzing ECG for Conduction Abnormalities

Let’s imagine you’re a PhD student analyzing an ECG from the **MIT-BIH Arrhythmia Database** to detect conduction abnormalities (heart blocks, BBB, WPW) for an ML/DL project. You’ll preprocess the signal, extract features (e.g., PR interval, QRS width), apply diagnostic rules, train a CNN to classify conditions, and visualize results.

### Step 1: Load Data (WFDB)
```python
import wfdb
import numpy as np

# Load ECG record (10 seconds, 360 Hz)
record = wfdb.rdrecord('mit-bih/100', sampto=10000)
signal = record.p_signal[:, 0]  # Lead I
fs = record.fs  # 360 Hz
annotation = wfdb.rdann('mit-bih/100', 'atr', sampto=10000)
ann_indices = annotation.sample
ann_labels = annotation.symbol
```

**What’s Happening**: We load a 10-second ECG from MIT-BIH, which includes annotations for QRS peaks and beat types. Lead I is used for simplicity.

**Analogy**: This is like opening a traffic log (ECG) with notes (annotations) marking key signals (QRS) and issues (abnormalities).

### Step 2: Preprocess Signal (SciPy)
```python
from scipy.signal import butter, filtfilt, iirnotch

def preprocess_ecg(data, fs):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40 / nyq
    b, a = butter(4, [low, high], btype='band')
    data_filt = filtfilt(b, a, data)
    b, a = iirnotch(60, 30, fs)
    return filtfilt(b, a, data_filt)

signal_clean = preprocess_ecg(signal, fs)
```

**What’s Happening**: We filter out baseline wander (<0.5 Hz), muscle noise (>40 Hz), and 60 Hz interference to enhance P, QRS, and T waves.

**Analogy**: This is like clearing road signs (ECG waves) of fog (noise) to see traffic flow (signals) clearly.

### Step 3: Fiducial Point Detection (NeuroKit2)
```python
import neurokit2 as nk

# Detect P, QRS waves
ecg_signals, info = nk.ecg_process(signal_clean, sampling_rate=fs)
p_peaks = info['ECG_P_Peaks']
r_peaks = info['ECG_R_Peaks']
q_peaks = info['ECG_Q_Peaks']
s_peaks = info['ECG_S_Peaks']
```

**What’s Happening**: `neurokit2` locates P (atrial activation), R (ventricular peak), Q (QRS start), and S (QRS end) for interval and morphology analysis.

**Analogy**: This is like marking traffic signals (R peaks) and stop signs (P, Q, S) on a highway (ECG).

### Step 4: Feature Extraction
```python
# RR intervals and heart rate
rr_intervals = np.diff(r_peaks) / fs
heart_rate = 60 / np.mean(rr_intervals)

# PR intervals and QRS widths
pr_intervals = [(q - p) / fs for p, q in zip(p_peaks, q_peaks) if not (np.isnan(p) or np.isnan(q))]
qrs_widths = [(s - q) / fs for q, s in zip(q_peaks, s_peaks) if not (np.isnan(q) or np.isnan(s))]

# P-to-QRS ratio
p_to_qrs_ratio = len(p_peaks) / len(r_peaks) if len(r_peaks) > 0 else 0

print(f"Heart Rate: {heart_rate:.1f} BPM, PR Mean: {np.mean(pr_intervals):.3f} s")
print(f"QRS Mean: {np.mean(qrs_widths):.3f} s, P-to-QRS Ratio: {p_to_qrs_ratio:.2f}")
```

**What’s Happening**: We extract PR intervals (AV conduction time), QRS widths (ventricular conduction), heart rate, and P-to-QRS ratio to detect blocks, BBB, or WPW.

**Analogy**: This is like measuring travel time (PR), road width (QRS), and traffic volume (P-to-QRS) to spot delays or detours.

### Step 5: Diagnostic Rules
```python
diagnoses = []

# First-Degree Heart Block
if np.mean(pr_intervals) > 0.2:
    diagnoses.append('First-Degree Heart Block')

# Second-Degree Heart Block
if p_to_qrs_ratio > 1.0:
    pr_diffs = np.diff(pr_intervals)
    if np.any(pr_diffs > 0.01):  # Progressive PR lengthening
        diagnoses.append('Second-Degree Type I (Wenckebach)')
    else:
        diagnoses.append('Second-Degree Type II')

# Third-Degree Heart Block
if p_to_qrs_ratio > 1.0 and heart_rate < 40:
    diagnoses.append('Third-Degree Heart Block')

# Bundle Branch Block
if np.mean(qrs_widths) > 0.12:
    diagnoses.append('Bundle Branch Block (RBBB or LBBB)')

# WPW Syndrome
if np.mean(pr_intervals) < 0.12 and np.mean(qrs_widths) > 0.12:
    diagnoses.append('WPW Syndrome')

print(f"Diagnoses: {diagnoses if diagnoses else 'No Conduction Abnormalities'}")
```

**What’s Happening**: Rules check PR prolongation (first-degree), dropped QRS (second-degree), P-QRS dissociation (third-degree), wide QRS (BBB), and short PR with wide QRS (WPW). These align with clinical criteria.

**Analogy**: This is like a traffic cop checking for slow roads (PR >200 ms), missing cars (dropped QRS), detours (wide QRS), or shortcuts (short PR, delta wave).

### Step 6: Deep Learning Model (TensorFlow)
We’ll train a CNN to classify ECG segments as **Normal**, **Heart Block**, **BBB**, or **WPW**, using segmented ECGs around R peaks.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Segment ECG (300 ms window, ~108 samples at 360 Hz)
segments = []
labels = []
window_size = 108
for i, r in enumerate(r_peaks):
    if r >= window_size//2 and r < len(signal_clean) - window_size//2:
        segment = signal_clean[r - window_size//2:r + window_size//2]
        if len(segment) == window_size:
            segments.append(segment)
            # Placeholder labels: simulate MIT-BIH labels
            if i < len(pr_intervals) and pr_intervals[i] > 0.2:
                labels.append(1)  # Heart Block
            elif i < len(qrs_widths) and qrs_widths[i] > 0.12:
                labels.append(2)  # BBB
            elif i < len(pr_intervals) and pr_intervals[i] < 0.12 and qrs_widths[i] > 0.12:
                labels.append(3)  # WPW
            else:
                labels.append(0)  # Normal

# Prepare data
X = np.array(segments)[:, :, np.newaxis]  # Shape: (n_segments, window_size, 1)
y = tf.keras.utils.to_categorical(labels)  # One-hot encode

# Build CNN
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size, 1)),
    Conv1D(64, 5, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes: Normal, Heart Block, BBB, WPW
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Predict on first 5 segments
predictions = model.predict(X[:5])
pred_labels = ['Normal', 'Heart Block', 'BBB', 'WPW']
print("Predictions:", [pred_labels[np.argmax(p)] for p in predictions])
```

**What’s Happening**: We segment ECGs into 300-ms windows, assign labels based on PR/QRS features (simulated for demo), and train a CNN to classify segments. The CNN learns patterns like prolonged PR (heart block), wide QRS (BBB), or delta waves (WPW). In practice, use annotated datasets like PTB-XL.

**Analogy**: The CNN is like a traffic engineer learning to spot slow roads (heart block), lane closures (BBB), or shortcuts (WPW) by studying traffic logs (ECG segments).

### Step 7: Visualize Results (Matplotlib)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(signal_clean, label='Filtered ECG', alpha=0.7)
plt.plot(r_peaks, signal_clean[r_peaks], 'ro', label='R Peaks')
plt.plot(p_peaks, signal_clean[p_peaks], 'go', label='P Peaks')
# Annotate diagnoses
for i, r in enumerate(r_peaks[:5]):
    if diagnoses:
        plt.text(r, signal_clean[r] + 0.1, diagnoses[min(i, len(diagnoses)-1)], fontsize=8)
plt.title('ECG with Detected Conduction Abnormalities')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.show()
```

**What’s Happening**: The plot shows the cleaned ECG with R peaks (red), P peaks (green), and text labels for detected abnormalities, visualizing rule-based and ML/DL results.

**Analogy**: This is like marking a traffic map with signals (R peaks), stop signs (P peaks), and notes (diagnoses) to show where traffic jams occur.

### Step 8: Summarize
- **Findings**: We loaded an ECG, preprocessed it, detected fiducial points, extracted PR/QRS features, applied diagnostic rules to identify heart blocks, BBB, and WPW, and trained a CNN for classification. The visualization confirmed detected abnormalities.
- **Outcome**: The pipeline produces features and classifications for ML/DL research, suitable for real-time monitoring or large-scale studies.
- **Next Steps**:
  - Use PTB-XL for multi-lead analysis (e.g., V1 for RBBB, V6 for LBBB).
  - Balance classes (e.g., oversample WPW) for better CNN performance.
  - Implement real-time detection for pacemakers or wearables.
  - Explore advanced DL (e.g., transformers for multi-lead patterns).

## Tips for PhD Preparation
- **Practice**: Download MIT-BIH or PTB-XL ECGs from PhysioNet and run this example. Try records with blocks (e.g., MIT-BIH 231 for heart block).
- **Visualize**: Plot ECGs with PR intervals and QRS widths to understand patterns.
- **Analogies**: Recall heart block as a slow/congested/collapsed road, BBB as a lane closure, and WPW as a shortcut.
- **ML/DL Focus**:
  - Use PTB-XL for conduction abnormality classification.
  - Experiment with SVMs for PR/QRS features vs. CNNs for raw ECG.
  - Study PhysioNet/CinC papers for ECG algorithms.
- **Tools**: Master `wfdb`, `neurokit2`, `scipy`, `tensorflow`. Explore `pywavelets` for QRS denoising.
- **Research Ideas**:
  - Real-time heart block detection in wearables.
  - Differentiating LBBB from MI using multi-lead features.
  - Predicting WPW tachycardia onset with sequence models.
