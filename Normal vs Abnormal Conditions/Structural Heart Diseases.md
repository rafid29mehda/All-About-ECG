# Structural Heart Diseases for ECG Biomedical Signal Processing

## Introduction to Structural Heart Diseases
**Structural heart diseases** involve changes to the heart’s anatomy—its chambers, walls, valves, or linings—impacting its electrical and mechanical function. These conditions, like **LVH/RVH**, **Atrial Enlargement**, **HCM**, **DCM**, **Pericarditis**, and **Endocarditis**, produce distinct ECG patterns critical for ML/DL analysis. They’re like architectural flaws in a house (heart), affecting how electricity (signals) flows. Below, I’ll explain each condition in a beginner-friendly way, tailored for your PhD journey.

---

## 1. Left or Right Ventricular Hypertrophy (LVH, RVH)

### 1.1 What is Ventricular Hypertrophy?
Ventricular hypertrophy is the thickening of the heart’s ventricular walls due to increased workload. **LVH** affects the left ventricle, **RVH** the right. It’s like a house with overly thick walls to handle extra pressure, but it strains the wiring (electrical system).

- **LVH**: Left ventricle thickens, often from high systemic pressure.
- **RVH**: Right ventricle thickens, often from high pulmonary pressure.

### 1.2 Physiology
- **Cause**:
  - **LVH**: Hypertension, aortic stenosis, hypertrophic cardiomyopathy, obesity.
  - **RVH**: Pulmonary hypertension, chronic lung disease (COPD), pulmonary embolism, congenital heart defects.
- **Effect**:
  - **LVH**: Increases cardiac workload, risks heart failure, arrhythmias, ischemia.
  - **RVH**: Strains right heart, risks right heart failure, arrhythmias.
- **Risk Factors**: Hypertension, obesity, smoking, lung disease, congenital defects.
- **Complications**: Heart failure, atrial fibrillation, sudden cardiac death.

### 1.3 ECG Features
- **LVH**:
  - **Increased QRS Amplitude**: Tall R waves in I, aVL, V5–V6; deep S waves in V1–V3.
  - **Sokolow-Lyon Criteria**: S in V1 + R in V5/V6 >35 mm, or R in aVL >11 mm.
  - **Left Axis Deviation**: QRS axis <-30°.
  - **ST-T Changes**: ST depression, T wave inversion in I, aVL, V5–V6 (strain pattern).
  - **Prolonged QRS**: Slightly wide (100–120 ms).
- **RVH**:
  - **Right Axis Deviation**: QRS axis >+90°.
  - **Tall R Waves in V1**: R/S ratio >1 in V1, or R >7 mm.
  - **Deep S Waves in V5–V6**: Reflects right ventricular dominance.
  - **ST-T Changes**: ST depression, T inversion in V1–V3.
- **Leads**: V1–V3 (RVH), V5–V6 (LVH), I, aVL (LVH axis).

### 1.4 How to Detect
- **Manual Analysis**:
  - **LVH**: Check QRS amplitude (Sokolow-Lyon), axis, ST-T strain.
  - **RVH**: Check R/S ratio in V1, right axis deviation, ST-T in V1–V3.
- **Automated Detection**:
  - **Features**: QRS amplitude, axis, ST-T deviation, R/S ratio.
  - **Algorithms**: Peak detection, axis calculation.
  - **Libraries**: `neurokit2` for QRS/ST, `scipy` for amplitude.
  - **ML/DL**: CNNs for QRS/ST patterns, multi-lead analysis.
- **Challenges**: Noise affects amplitude; LVH mimics MI, RVH mimics RBBB.

### 1.5 How to Solve
- **Clinical**:
  - **LVH**: Control hypertension (ACE inhibitors, beta-blockers), treat aortic stenosis, lifestyle (weight loss).
  - **RVH**: Manage pulmonary hypertension (vasodilators), treat lung disease, oxygen therapy.
  - **Both**: Monitor, treat arrhythmias, consider ICD for high risk.
- **Signal Processing**:
  - Enhance QRS/ST with band-pass filter (0.5–40 Hz).
  - Extract QRS amplitude/axis for ML/DL models.
  - Develop hypertrophy detection for risk stratification.

### 1.6 Example
Detect LVH:
```python
import neurokit2 as nk
import numpy as np
signal = np.random.rand(5000)  # Placeholder ECG
fs = 500
ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
r_peaks = info['ECG_R_Peaks']
r_amplitudes = [signal[r] for r in r_peaks]
if np.mean(r_amplitudes) > 3.5:  # Simplified Sokolow-Lyon
    print("Possible Left Ventricular Hypertrophy")
```

**Analogy**: LVH/RVH is like a house with thickened walls (ventricles), straining the wiring (QRS amplitude) and lights (ST-T changes).

---

## 2. Atrial Enlargement

### 2.1 What is Atrial Enlargement?
Atrial enlargement is the stretching or thickening of the atria due to increased pressure or volume. **Left Atrial Enlargement (LAE)** and **Right Atrial Enlargement (RAE)** occur independently or together (biatrial). It’s like expanding a house’s upstairs rooms, messing with the electrical outlets (P waves).

### 2.2 Physiology
- **Cause**:
  - **LAE**: Mitral valve disease, hypertension, heart failure, atrial fibrillation.
  - **RAE**: Pulmonary hypertension, tricuspid valve disease, COPD, congenital defects.
- **Effect**:
  - **LAE**: Risks atrial fibrillation, stroke, reduced cardiac output.
  - **RAE**: Risks right heart failure, arrhythmias.
- **Risk Factors**: Hypertension, valvular disease, lung disease, obesity.
- **Complications**: Atrial fibrillation, thromboembolism, heart failure.

### 2.3 ECG Features
- **LAE**:
  - **Wide P Wave**: >120 ms in II, notched (“P mitrale”).
  - **Negative P Terminal Force in V1**: >0.04 mm·s (deep, prolonged negative phase).
  - **Increased P Amplitude**: >2.5 mm in II (less common).
- **RAE**:
  - **Peaked P Wave**: >2.5 mm in II, III, aVF (“P pulmonale”).
  - **Positive P in V1**: >1.5 mm.
  - **Right Axis Deviation**: Rare, if severe.
- **Leads**: II, V1 (LAE negative, RAE positive), III, aVF (RAE).

### 2.4 How to Detect
- **Manual Analysis**:
  - **LAE**: Check P width in II, negative P terminal in V1.
  - **RAE**: Check P height in II, positive P in V1.
- **Automated Detection**:
  - **Features**: P wave duration, amplitude, terminal force.
  - **Algorithms**: P wave detection, area calculation.
  - **Libraries**: `neurokit2` for P waves, `scipy` for area.
  - **ML/DL**: CNNs for P wave patterns.
- **Challenges**: Noise obscures P waves; AF masks atrial enlargement.

### 2.5 How to Solve
- **Clinical**:
  - **LAE**: Treat hypertension, manage mitral disease, prevent AF (antiarrhythmics).
  - **RAE**: Treat pulmonary hypertension, manage tricuspid disease, oxygen therapy.
  - **Both**: Monitor, anticoagulation for AF risk, lifestyle changes.
- **Signal Processing**:
  - Enhance P waves with high-pass filter (5 Hz).
  - Extract P wave features for ML/DL models.
  - Develop AF risk prediction systems.

### 2.6 Example
Detect LAE:
```python
p_peaks = info['ECG_P_Peaks']
p_durations = [(p_end - p_start) / fs for p_start, p_end in zip(p_peaks[:-1], p_peaks[1:])]
if np.mean(p_durations) > 0.12:
    print("Possible Left Atrial Enlargement")
```

**Analogy**: Atrial enlargement is like oversized upstairs rooms (atria), stretching the wiring (P waves) wider (LAE) or taller (RAE).

---

## 3. Hypertrophic Cardiomyopathy (HCM)

### 3.1 What is HCM?
HCM is a genetic condition where the heart muscle (myocardium) thickens abnormally, often in the left ventricle, obstructing blood flow. It’s like a house with overly thick internal walls, blocking hallways (blood flow) and straining wiring.

### 3.2 Physiology
- **Cause**: Genetic mutations (e.g., sarcomere genes), leading to myofibril disarray.
- **Effect**: Left ventricular outflow obstruction, diastolic dysfunction, ischemia, arrhythmias.
- **Risk Factors**: Family history, young athletes, genetic predisposition.
- **Complications**: Sudden cardiac death, heart failure, atrial fibrillation.

### 3.3 ECG Features
- **LVH Patterns**: Tall R in V5–V6, deep S in V1–V3 (Sokolow-Lyon).
- **Pathological Q Waves**: >40 ms or >25% R height in I, aVL, V5–V6 (mimic MI).
- **ST-T Changes**: ST depression, T wave inversion in I, aVL, V4–V6.
- **Left Axis Deviation**: QRS axis <-30°.
- **Arrhythmias**: AF, VT, PVCs common.
- **Leads**: V5–V6 (LVH, Q waves), I, aVL (ST-T).

### 3.4 How to Detect
- **Manual Analysis**:
  - Check LVH criteria, pathological Q waves, ST-T changes, arrhythmias.
  - Confirm with echocardiogram (thick septum).
- **Automated Detection**:
  - **Features**: QRS amplitude, Q wave duration, ST-T deviation, arrhythmia frequency.
  - **Algorithms**: QRS/ST analysis, arrhythmia detection.
  - **Libraries**: `neurokit2` for QRS/ST, `scipy` for Q waves.
  - **ML/DL**: CNNs for multi-feature patterns, LSTMs for arrhythmias.
- **Challenges**: Mimics MI, athlete’s heart; variable ECG findings.

### 3.5 How to Solve
- **Clinical**:
  - **Symptomatic**: Beta-blockers, calcium channel blockers, septal myectomy, alcohol ablation.
  - **Arrhythmias**: Antiarrhythmics, ICD for sudden death risk.
  - **Screening**: Genetic testing, family screening.
- **Signal Processing**:
  - Enhance QRS/ST with band-pass filter (0.5–40 Hz).
  - Extract Q/ST features for ML/DL models.
  - Develop sudden death risk predictors.

### 3.6 Example
Detect HCM:
```python
q_amplitudes = [np.min(signal[q-10:q+10]) for q in info['ECG_Q_Peaks'] if not np.isnan(q)]
if np.any(np.array(q_amplitudes) < -0.25 * np.mean(r_amplitudes)):
    print("Possible Hypertrophic Cardiomyopathy (Pathological Q Waves)")
```

**Analogy**: HCM is like a house with thick, chaotic walls (myocardium), blocking doors (outflow) and sparking wires (Q waves, arrhythmias).

---

## 4. Dilated Cardiomyopathy (DCM)

### 4.1 What is DCM?
DCM is a condition where the heart’s ventricles dilate and weaken, reducing pumping efficiency. It’s like a house with overstretched, thin walls, leaking electricity (signals).

### 4.2 Physiology
- **Cause**: Genetics, alcohol, viral myocarditis, toxins, pregnancy (peripartum).
- **Effect**: Systolic dysfunction, heart failure, arrhythmias, thromboembolism.
- **Risk Factors**: Family history, alcoholism, viral infections, chemotherapy.
- **Complications**: Heart failure, VT/VF, stroke, sudden death.

### 4.3 ECG Features
- **Low QRS Voltage**: <5 mm in limb leads, <10 mm in precordial leads (dilated chambers).
- **Poor R Wave Progression**: Small or absent R waves in V1–V3.
- **LBBB**: Wide QRS (>120 ms), notched R in V5–V6, common in DCM.
- **Arrhythmias**: AF, PVCs, VT frequent.
- **ST-T Changes**: Non-specific ST depression, T inversion.
- **Leads**: V1–V3 (poor R progression), V5–V6 (LBBB).

### 4.4 How to Detect
- **Manual Analysis**:
  - Check low QRS voltage, poor R progression, LBBB, arrhythmias.
  - Confirm with echocardiogram (dilated ventricles).
- **Automated Detection**:
  - **Features**: QRS amplitude, R progression, QRS width, arrhythmia frequency.
  - **Algorithms**: QRS analysis, arrhythmia detection.
  - **Libraries**: `neurokit2` for QRS, `scipy` for amplitude.
  - **ML/DL**: CNNs for QRS/arrhythmia patterns.
- **Challenges**: Low voltage mimics pericardial effusion; LBBB obscures MI.

### 4.5 How to Solve
- **Clinical**:
  - **Heart Failure**: ACE inhibitors, beta-blockers, diuretics, CRT.
  - **Arrhythmias**: Antiarrhythmics, ICD for VT/VF risk.
  - **Transplant**: For end-stage DCM.
- **Signal Processing**:
  - Enhance QRS with band-pass filter (0.5–40 Hz).
  - Extract QRS/arrhythmia features for ML/DL models.
  - Develop heart failure risk predictors.

### 4.6 Example
Detect DCM:
```python
if np.mean(r_amplitudes) < 0.5:
    print("Possible Dilated Cardiomyopathy (Low QRS Voltage)")
```

**Analogy**: DCM is like a house with stretched, weak walls (ventricles), dimming lights (low QRS) and sparking wires (arrhythmias).

---

## 5. Pericarditis (Inflammation of the Pericardium)

### 5.1 What is Pericarditis?
Pericarditis is inflammation of the pericardium (heart’s outer sac), often causing chest pain. It’s like a house with an irritated outer layer, affecting the wiring inside.

### 5.2 Physiology
- **Cause**: Viral infections, autoimmune diseases, MI (post-infarction), uremia, trauma.
- **Effect**: Pericardial friction, effusion (fluid buildup), pain, rarely tamponade.
- **Risk Factors**: Viral illness, autoimmune conditions, kidney failure, recent MI.
- **Complications**: Pericardial effusion, tamponade, chronic pericarditis.

### 5.3 ECG Features
- **Diffuse ST Elevation**: Concave upward, in most leads (I, II, aVL, aVF, V2–V6).
- **PR Depression**: >0.5 mm, especially in II, aVF, V4–V6 (atrial inflammation).
- **Low QRS Voltage**: If effusion present.
- **Electrical Alternans**: Alternating QRS amplitude (severe effusion/tamponade).
- **No Reciprocal ST Depression**: Unlike MI.
- **Leads**: II, aVF, V4–V6 (ST/PR changes).

### 5.4 How to Detect
- **Manual Analysis**:
  - Look for diffuse ST elevation, PR depression, low voltage if effusion.
  - Confirm with clinical symptoms (chest pain, fever), echocardiogram.
- **Automated Detection**:
  - **Features**: ST elevation, PR depression, QRS amplitude, alternans.
  - **Algorithms**: ST/PR analysis, amplitude variation.
  - **Libraries**: `neurokit2` for ST/PR, `scipy` for alternans.
  - **ML/DL**: CNNs for ST/PR patterns.
- **Challenges**: Mimics MI; subtle PR changes.

### 5.5 How to Solve
- **Clinical**:
  - **Acute**: NSAIDs (ibuprofen), colchicine, corticosteroids for severe cases.
  - **Effusion/Tamponade**: Pericardiocentesis, treat cause.
  - **Monitor**: ECG, echo for complications.
- **Signal Processing**:
  - Enhance ST/PR with band-pass filter (0.5–40 Hz).
  - Extract ST/PR features for ML/DL models.
  - Develop tamponade detection systems.

### 5.6 Example
Detect pericarditis:
```python
st_values = [signal_clean[r + int(0.1 * fs)] - np.mean(signal_clean[r-50:r-10]) for r in r_peaks]
if np.mean(st_values) > 0.1 and np.any([signal_clean[p-10] < -0.05 for p in p_peaks]):
    print("Possible Pericarditis (ST Elevation, PR Depression)")
```

**Analogy**: Pericarditis is like a house with a scratched outer paint (pericardium), sparking wires (ST elevation) and dim outlets (PR depression).

---

## 6. Endocarditis (Infection of the Heart Valves)

### 6.1 What is Endocarditis?
Endocarditis is an infection of the heart’s inner lining, usually valves, causing inflammation and damage. It’s like a house with mold (infection) on its plumbing (valves), disrupting flow and wiring.

### 6.2 Physiology
- **Cause**: Bacterial (e.g., Streptococcus, Staphylococcus), fungal, often from dental procedures, IV drug use, prosthetic valves.
- **Effect**: Valve damage, vegetations (clots), embolism, heart failure, arrhythmias.
- **Risk Factors**: Valvular disease, prosthetic valves, IV drug use, dental issues.
- **Complications**: Stroke, heart failure, abscess, septicemia.

### 6.3 ECG Features
- **Non-Specific**: No direct ECG hallmark, reflects complications.
- **Conduction Abnormalities**: AV block, bundle branch block (abscess near conduction system).
- **Arrhythmias**: AF, PVCs, VT if myocardium involved.
- **Low QRS Voltage**: If pericardial effusion from inflammation.
- **ST-T Changes**: Non-specific, may mimic ischemia.
- **Leads**: II (conduction), V1–V3 (blocks), diffuse (effusion).

### 6.4 How to Detect
- **Manual Analysis**:
  - Look for AV block, arrhythmias, low voltage, non-specific ST-T.
  - Confirm with blood cultures, echocardiogram (vegetations).
- **Automated Detection**:
  - **Features**: PR interval, QRS amplitude, arrhythmia frequency, ST-T deviation.
  - **Algorithms**: Conduction analysis, arrhythmia detection.
  - **Libraries**: `neurokit2` for PR/QRS, `scipy` for ST.
  - **ML/DL**: CNNs for conduction/arrhythmia patterns.
- **Challenges**: Non-specific ECG; requires clinical correlation.

### 6.5 How to Solve
- **Clinical**:
  - **Acute**: IV antibiotics (4–6 weeks), surgery for valve damage or abscess.
  - **Prevention**: Antibiotic prophylaxis for high-risk patients (e.g., dental procedures).
  - **Monitor**: ECG, echo for complications.
- **Signal Processing**:
  - Enhance QRS/ST with band-pass filter (0.5–40 Hz).
  - Extract conduction/arrhythmia features for ML/DL models.
  - Develop complication detection systems.

### 6.6 Example
Detect endocarditis complications:
```python
pr_intervals = [(q - p) / fs for p, q in zip(p_peaks, q_peaks) if not (np.isnan(p) or np.isnan(q))]
if np.mean(pr_intervals) > 0.2:
    print("Possible Endocarditis (AV Block)")
```

**Analogy**: Endocarditis is like moldy plumbing (valves), causing electrical flickers (blocks, arrhythmias) in the house.

---

## End-to-End Example: Analyzing ECG for Structural Heart Diseases

Let’s imagine you’re a PhD student analyzing an ECG from the **PTB-XL Database** to detect structural heart diseases (LVH/RVH, atrial enlargement, HCM, DCM, pericarditis, endocarditis) for an ML/DL project. You’ll preprocess the signal, extract features (e.g., QRS amplitude, P wave), apply diagnostic rules, train a CNN to classify conditions, and visualize results.

### Step 1: Load Data (WFDB)
```python
import wfdb
import numpy as np

# Load ECG record (10 seconds, 500 Hz)
record = wfdb.rdrecord('ptb-xl/00100_hr', sampto=5000)
signal = record.p_signal[:, 1]  # Lead II
fs = record.fs  # 500 Hz
```

**What’s Happening**: We load a 10-second ECG from PTB-XL, a dataset with labeled cardiac conditions. Lead II is chosen for clear P, QRS, and T waves.

**Analogy**: This is like opening a house blueprint (ECG) to check for structural flaws (heart diseases).

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

**What’s Happening**: We filter out baseline wander (<0.5 Hz), muscle noise (>40 Hz), and 60 Hz interference to enhance P, QRS, and ST.

**Analogy**: This is like clearing dust from a blueprint to see walls (QRS) and wiring (P/ST) clearly.

### Step 3: Fiducial Point Detection (NeuroKit2)
```python
import neurokit2 as nk

# Detect P, QRS, T waves
ecg_signals, info = nk.ecg_process(signal_clean, sampling_rate=fs)
p_peaks = info['ECG_P_Peaks']
r_peaks = info['ECG_R_Peaks']
q_peaks = info['ECG_Q_Peaks']
s_peaks = info['ECG_S_Peaks']
t_peaks = info['ECG_T_Peaks']
```

**What’s Happening**: `neurokit2` locates P (atrial), R (ventricular), Q/S (QRS boundaries), and T (repolarization) for feature extraction.

**Analogy**: This is like marking doors (R peaks), windows (P peaks), and outlets (T peaks) on a blueprint.

### Step 4: Feature Extraction
```python
# RR intervals and heart rate
rr_intervals = np.diff(r_peaks) / fs
heart_rate = 60 / np.mean(rr_intervals)

# QRS amplitudes, P wave durations, ST deviations
r_amplitudes = [signal_clean[r] for r in r_peaks if not np.isnan(r)]
q_amplitudes = [np.min(signal_clean[int(q-10):int(q+10)]) for q in q_peaks if not np.isnan(q)]
p_durations = [(p_end - p_start) / fs for p_start, p_end in zip(p_peaks[:-1], p_peaks[1:]) if not (np.isnan(p_start) or np.isnan(p_end))]
st_values = [signal_clean[r + int(0.1 * fs)] - np.mean(signal_clean[r-50:r-10]) for r in r_peaks]
pr_intervals = [(q - p) / fs for p, q in zip(p_peaks, q_peaks) if not (np.isnan(p) or np.isnan(q))]
qrs_widths = [(s - q) / fs for q, s in zip(q_peaks, s_peaks) if not (np.isnan(q) or np.isnan(s))]

print(f"Heart Rate: {heart_rate:.1f} BPM, QRS Amplitude: {np.mean(r_amplitudes):.3f} mV")
print(f"P Duration: {np.mean(p_durations):.3f} s, ST Mean: {np.mean(st_values):.3f} mV")
```

**What’s Happening**: We extract QRS amplitude (LVH/RVH), P duration (atrial enlargement), ST deviation (pericarditis), Q waves (HCM), QRS width (DCM), and PR intervals (endocarditis) to detect structural diseases.

**Analogy**: This is like measuring wall thickness (QRS), room size (P duration), and wire strain (ST) in a house.

### Step 5: Diagnostic Rules
```python
diagnoses = []

# LVH
if np.mean(r_amplitudes) > 3.5:  # Sokolow-Lyon
    diagnoses.append('Left Ventricular Hypertrophy')

# RVH (simplified)
if np.mean(r_amplitudes) > 0.7 and heart_rate > 100:  # V1 R >7 mm
    diagnoses.append('Right Ventricular Hypertrophy')

# Atrial Enlargement
if np.mean(p_durations) > 0.12:
    diagnoses.append('Left Atrial Enlargement')
elif np.max([signal_clean[p] for p in p_peaks if not np.isnan(p)]) > 0.25:
    diagnoses.append('Right Atrial Enlargement')

# HCM
if np.any(np.array(q_amplitudes) < -0.25 * np.mean(r_amplitudes)):
    diagnoses.append('Hypertrophic Cardiomyopathy')

# DCM
if np.mean(r_amplitudes) < 0.5 or np.mean(qrs_widths) > 0.12:
    diagnoses.append('Dilated Cardiomyopathy')

# Pericarditis
if np.mean(st_values) > 0.1 and np.any([signal_clean[p-10] < -0.05 for p in p_peaks if not np.isnan(p)]):
    diagnoses.append('Pericarditis')

# Endocarditis
if np.mean(pr_intervals) > 0.2:
    diagnoses.append('Endocarditis (AV Block)')

print(f"Diagnoses: {diagnoses if diagnoses else 'No Structural Heart Diseases'}")
```

**What’s Happening**: Rules check QRS amplitude (LVH/RVH), P duration/amplitude (atrial enlargement), Q waves (HCM), low QRS/LBBB (DCM), ST/PR changes (pericarditis), and PR prolongation (endocarditis). These align with clinical criteria.

**Analogy**: This is like an architect checking thick walls (LVH), big rooms (atrial enlargement), blocked doors (HCM), weak walls (DCM), scratched paint (pericarditis), or moldy pipes (endocarditis).

### Step 6: Deep Learning Model (TensorFlow)
We’ll train a CNN to classify ECG segments as **Normal**, **Hypertrophy (LVH/RVH)**, **Cardiomyopathy (HCM/DCM)**, or **Inflammation (Pericarditis/Endocarditis)**, using segmented ECGs around R peaks.

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
            # Placeholder labels: simulate PTB-XL
            if i < len(r_amplitudes) and (r_amplitudes[i] > 3.5 or r_amplitudes[i] > 0.7):
                labels.append(1)  # Hypertrophy
            elif i < len(q_amplitudes) and (q_amplitudes[i] < -0.25 * r_amplitudes[i] or r_amplitudes[i] < 0.5):
                labels.append(2)  # Cardiomyopathy
            elif i < len(st_values) and (st_values[i] > 0.1 or pr_intervals[i] > 0.2):
                labels.append(3)  # Inflammation
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
    Dense(4, activation='softmax')  # 4 classes
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Predict on first 5 segments
predictions = model.predict(X[:5])
pred_labels = ['Normal', 'Hypertrophy', 'Cardiomyopathy', 'Inflammation']
print("Predictions:", [pred_labels[np.argmax(p)] for p in predictions])
```

**What’s Happening**: We segment ECGs into 300-ms windows, assign labels based on QRS/P/ST features (simulated), and train a CNN to classify segments. The CNN learns patterns like high QRS (hypertrophy), Q waves (HCM), or ST elevation (pericarditis). Use PTB-XL labels in practice.

**Analogy**: The CNN is like an architect learning to spot thick walls (hypertrophy), weak structures (cardiomyopathy), or damaged exteriors (inflammation) in a blueprint.

### Step 7: Visualize Results (Matplotlib)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(signal_clean, label='Filtered ECG', alpha=0.7)
plt.plot(r_peaks, signal_clean[r_peaks], 'ro', label='R Peaks')
plt.plot(p_peaks, signal_clean[p_peaks], 'go', label='P Peaks')
plt.plot(t_peaks, signal_clean[t_peaks], 'bo', label='T Peaks')
# Annotate diagnoses
for i, r in enumerate(r_peaks[:5]):
    if diagnoses:
        plt.text(r, signal_clean[r] + 0.1, diagnoses[min(i, len(diagnoses)-1)], fontsize=8)
plt.title('ECG with Detected Structural Heart Diseases')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.show()
```

**What’s Happening**: The plot shows the cleaned ECG with R peaks (red), P peaks (green), T peaks (blue), and text labels for detected diseases, visualizing rule-based and ML/DL results.

**Analogy**: This is like marking a blueprint with walls (R peaks), rooms (P peaks), outlets (T peaks), and repair notes (diagnoses).

### Step 8: Summarize
- **Findings**: We loaded an ECG, preprocessed it, detected fiducial points, extracted QRS/P/ST features, applied diagnostic rules to identify structural heart diseases, and trained a CNN for classification. The visualization confirmed detected abnormalities.
- **Outcome**: The pipeline produces features and classifications for ML/DL research, suitable for clinical diagnostics or large-scale studies.
- **Next Steps**:
  - Use PTB-XL for 12-lead analysis (e.g., V1 for RVH, V6 for LVH).
  - Balance classes (e.g., oversample HCM) for better CNN performance.
  - Implement real-time detection for echo-ECG integration.
  - Explore advanced DL (e.g., transformers for multi-lead patterns).

## Tips for PhD Preparation
- **Practice**: Download PTB-XL or MIT-BIH ECGs from PhysioNet and run this example. Try records with LVH (PTB-XL 00101_hr).
- **Visualize**: Plot ECGs with QRS, P, and ST to understand patterns.
- **Analogies**: Recall LVH/RVH as thick walls, atrial enlargement as big rooms, HCM as chaotic walls, DCM as weak walls, pericarditis as scratched plaster, endocarditis as moldy pipes.
- **ML/DL Focus**:
  - Use PTB-XL for structural disease classification.
  - Experiment with SVMs for QRS/P features vs. CNNs for raw ECG.
  - Study PhysioNet/CinC papers for ECG algorithms.
- **Tools**: Master `wfdb`, `neurokit2`, `scipy`, `tensorflow`. Explore `pywavelets` for QRS/ST denoising.
- **Research Ideas**:
  - Real-time LVH detection in hypertension patients.
  - Differentiating HCM vs. athlete’s heart with ML.
  - Predicting heart failure in DCM using ECG and HRV.
