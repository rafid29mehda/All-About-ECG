# Arrhythmias for ECG Biomedical Signal Processing

## Introduction to Arrhythmias
**Arrhythmias** are abnormal heart rhythms where the heart beats too fast, too slow, or irregularly, disrupting its normal electrical activity. They’re like a band playing out of sync—sometimes chaotic, sometimes too slow, or with random extra notes. Understanding arrhythmias is crucial for ECG analysis, as they’re key targets for ML/DL models in diagnosing cardiac conditions. Below, I’ll break down each arrhythmia in detail, tailored for our PhD journey.

---

## 1. Atrial Fibrillation (AF)

### 1.1 What is Atrial Fibrillation?
AF is a common arrhythmia where the atria (upper heart chambers) quiver chaotically instead of contracting properly. It’s like a crowd of musicians playing random notes instead of a coordinated melody.

### 1.2 Physiology
- **Cause**: Multiple electrical impulses fire randomly in the atria, often from ectopic foci or re-entry circuits. Common triggers include atrial dilation, fibrosis, or inflammation.
- **Effect**: Ineffective atrial contraction reduces cardiac output by 10–20%. Blood pools in atria, increasing risk of clots and stroke.
- **Risk Factors**: Age >65, hypertension, heart disease (e.g., heart failure, MI), thyroid disorders, alcohol, sleep apnea.
- **Complications**: Stroke (5x risk), heart failure, fatigue.

### 1.3 ECG Features
- **No P Waves**: Replaced by irregular, low-amplitude fibrillatory (f) waves (4–10 Hz).
- **Irregular RR Intervals**: QRS complexes occur unpredictably due to erratic AV node conduction.
- **Normal QRS**: Narrow (<120 ms) unless conduction abnormalities (e.g., bundle branch block).
- **Rate**: Uncontrolled (100–180 bpm), controlled (<100 bpm with medication).
- **Leads**: Best seen in II, V1 (clear f waves).

### 1.4 How to Detect
- **Manual Analysis**:
  - Look for absent P waves, irregular RR intervals, and wavy baseline.
  - Confirm with multiple leads (II, V1).
- **Automated Detection**:
  - **Features**: RR interval variability (e.g., standard deviation, entropy), absence of P waves, frequency analysis (f waves).
  - **Algorithms**: Pan-Tompkins for QRS detection, wavelet transforms for f waves.
  - **Libraries**: `neurokit2` for RR analysis, `pywavelets` for frequency decomposition.
  - **ML/DL**: CNNs for raw ECG classification, LSTMs for RR sequence analysis.
- **Challenges**: Noise mimics f waves; irregular rhythms complicate QRS detection.

### 1.5 How to Solve
- **Clinical**:
  - **Rate Control**: Beta-blockers (e.g., metoprolol), calcium channel blockers (e.g., diltiazem).
  - **Rhythm Control**: Antiarrhythmics (e.g., amiodarone), cardioversion.
  - **Anticoagulation**: Warfarin or DOACs (e.g., apixaban) to prevent stroke.
  - **Ablation**: Catheter ablation to destroy ectopic foci.
- **Signal Processing**:
  - Denoise ECG (band-pass filter 0.5–40 Hz).
  - Extract RR intervals and entropy for ML/DL models.
  - Train models to detect AF in real-time (e.g., wearables).

### 1.6 Example
Analyze an ECG for AF using Python:
```python
import neurokit2 as nk
import numpy as np
signal = np.random.rand(5000)  # Placeholder ECG (replace with real data)
fs = 500  # Sampling rate
ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
rr_intervals = np.diff(info['ECG_R_Peaks']) / fs
rr_std = np.std(rr_intervals)
entropy = nk.entropy_approximate(rr_intervals)
print(f"RR Std: {rr_std:.3f} s, Entropy: {entropy:.3f}")
if rr_std > 0.1 and entropy > 1.0:
    print("Likely Atrial Fibrillation")
```

**Analogy**: AF is like a jazz band improvising chaotically, with no clear rhythm (no P waves, irregular RR).

---

## 2. Atrial Flutter

### 2.1 What is Atrial Flutter?
Atrial flutter is a fast, organized atrial rhythm where the atria contract rapidly (250–350 bpm) in a circular pattern. It’s like a metronome stuck on high speed, with the atria beating in a predictable loop.

### 2.2 Physiology
- **Cause**: Re-entry circuit in the atria (often right atrium), typically from scar tissue or atrial dilation.
- **Effect**: Atria contract too fast, but AV node blocks some impulses (e.g., 2:1 or 4:1 conduction), so ventricular rate is slower (e.g., 150 bpm).
- **Risk Factors**: Heart surgery, hypertension, COPD, thyroid disease.
- **Complications**: Stroke, heart failure, conversion to AF.

### 2.3 ECG Features
- **Flutter (F) Waves**: Sawtooth pattern, regular, at 250–350 bpm (best in II, III, aVF, V1).
- **Regular/Irregular QRS**: Depends on AV conduction (e.g., 2:1 → 150 bpm, regular; 3:1 → irregular).
- **No P Waves**: Replaced by F waves.
- **Normal QRS**: Narrow unless conduction issues.
- **Leads**: II, III, aVF show clear F waves.

### 2.4 How to Detect
- **Manual Analysis**:
  - Identify sawtooth F waves, count ventricular rate (e.g., 150 bpm → 2:1).
  - Check lead V1 for distinct F waves.
- **Automated Detection**:
  - **Features**: Frequency analysis (F waves at 4–6 Hz), RR regularity.
  - **Algorithms**: Fourier transform for F wave frequency, QRS detection for rate.
  - **Libraries**: `scipy.signal` for spectral analysis, `biosppy` for QRS.
  - **ML/DL**: CNNs to detect sawtooth patterns.
- **Challenges**: F waves may resemble noise; variable AV conduction complicates rhythm.

### 2.5 How to Solve
- **Clinical**:
  - **Rate Control**: Beta-blockers, calcium channel blockers.
  - **Rhythm Control**: Cardioversion, antiarrhythmics (e.g., ibutilide).
  - **Anticoagulation**: Similar to AF.
  - **Ablation**: Targets re-entry circuit (high success rate).
- **Signal Processing**:
  - Filter ECG (0.5–40 Hz) to enhance F waves.
  - Use frequency-domain features for ML/DL classification.

### 2.6 Example
Detect flutter waves:
```python
from scipy.signal import welch
freqs, psd = welch(signal, fs=fs, nperseg=1024)
if 4 < freqs[np.argmax(psd)] < 6:
    print("Possible Atrial Flutter (F waves detected)")
```

**Analogy**: Atrial flutter is like a carousel spinning too fast, with regular loops (F waves) but only some riders (QRS) reaching the ground.

---

## 3. Ventricular Tachycardia (VT)

### 3.1 What is Ventricular Tachycardia?
VT is a fast, life-threatening rhythm originating from the ventricles (>100 bpm). It’s like a rogue drummer pounding rapidly, ignoring the conductor (atria).

### 3.2 Physiology
- **Cause**: Ectopic focus or re-entry in ventricles, often from scar tissue (e.g., post-MI), cardiomyopathy, or electrolyte imbalance.
- **Effect**: Ineffective pumping, leading to low blood pressure, syncope, or cardiac arrest.
- **Risk Factors**: Prior MI, heart failure, long QT syndrome, drug toxicity.
- **Complications**: Can degenerate into VF, sudden cardiac death.

### 3.3 ECG Features
- **Wide QRS**: >120 ms, bizarre shape (monomorphic or polymorphic).
- **Regular Rhythm**: Consistent RR intervals (monomorphic VT).
- **No P Waves**: Or dissociated P waves (AV dissociation).
- **Fusion/Capture Beats**: Occasional normal QRS (capture) or mixed QRS (fusion).
- **Leads**: V1–V6 show wide, abnormal QRS patterns.

### 3.4 How to Detect
- **Manual Analysis**:
  - Look for ≥3 wide QRS beats at >100 bpm.
  - Check for AV dissociation, fusion beats.
- **Automated Detection**:
  - **Features**: QRS width, RR regularity, QRS morphology.
  - **Algorithms**: QRS detection, morphology analysis.
  - **Libraries**: `neurokit2` for QRS width, `wfdb` for annotations.
  - **ML/DL**: CNNs for QRS shape classification.
- **Challenges**: Differentiating VT from SVT with aberrant conduction.

### 3.5 How to Solve
- **Clinical**:
  - **Emergency**: Cardioversion, antiarrhythmics (e.g., amiodarone).
  - **Long-Term**: Implantable cardioverter-defibrillator (ICD), ablation.
  - **Treat Cause**: Correct electrolytes, manage heart disease.
- **Signal Processing**:
  - Enhance QRS with band-pass filter (5–40 Hz).
  - Train real-time ML/DL models for ICDs or monitors.

### 3.6 Example
Check QRS width:
```python
qrs_widths = [(s - q) / fs for q, s in zip(info['ECG_Q_Peaks'], info['ECG_S_Peaks'])]
if np.mean(qrs_widths) > 0.12 and heart_rate > 100:
    print("Possible Ventricular Tachycardia")
```

**Analogy**: VT is like a heavy metal band playing too fast and loud, drowning out the rest of the orchestra (wide QRS, no P waves).

---

## 4. Ventricular Fibrillation (VF)

### 4.1 What is Ventricular Fibrillation?
VF is a chaotic, life-threatening rhythm where the ventricles quiver ineffectively, stopping blood flow. It’s like static noise replacing the heart’s music, requiring immediate intervention.

### 4.2 Physiology
- **Cause**: Multiple re-entry circuits in ventricles, often from MI, ischemia, or electrocution.
- **Effect**: No cardiac output, leading to cardiac arrest within seconds.
- **Risk Factors**: MI, heart failure, electrolyte imbalances, drug toxicity.
- **Complications**: Death without defibrillation.

### 4.3 ECG Features
- **No Recognizable Waves**: Irregular, high-frequency (4–10 Hz) waves, no P, QRS, or T.
- **Chaotic Baseline**: Amplitudes vary, no pattern.
- **Leads**: All leads show disordered signal.

### 4.4 How to Detect
- **Manual Analysis**:
  - Identify absence of P, QRS, T; chaotic, irregular waves.
- **Automated Detection**:
  - **Features**: High entropy, broad frequency spectrum, no QRS.
  - **Algorithms**: Spectral analysis, entropy calculation.
  - **Libraries**: `scipy.signal` for PSD, `neurokit2` for entropy.
  - **ML/DL**: CNNs for chaos detection.
- **Challenges**: Noise or motion artifacts mimic VF.

### 4.5 How to Solve
- **Clinical**:
  - **Emergency**: Defibrillation, CPR, epinephrine.
  - **Long-Term**: ICD, treat underlying cause (e.g., ischemia).
- **Signal Processing**:
  - Denoise ECG to isolate chaotic signal.
  - Develop real-time VF detection for AEDs using ML/DL.

### 4.6 Example
Calculate entropy:
```python
entropy = nk.entropy_approximate(signal)
if entropy > 1.5 and not info['ECG_R_Peaks']:
    print("Possible Ventricular Fibrillation")
```

**Analogy**: VF is like a radio stuck on static, with no melody or rhythm, just noise (chaotic waves).

---

## 5. Bradycardia

### 5.1 What is Bradycardia?
Bradycardia is a slow heart rate (<60 bpm) that may impair blood flow. It’s like a band playing too slowly, making the audience (body) sluggish.

### 5.2 Physiology
- **Cause**: SA node dysfunction, AV block, medications (e.g., beta-blockers), high vagal tone (athletes), hypothyroidism.
- **Effect**: Reduced cardiac output, causing fatigue, dizziness, or syncope.
- **Risk Factors**: Aging, heart disease, electrolyte imbalances, sleep apnea.
- **Complications**: Syncope, heart failure, need for pacemaker.

### 5.3 ECG Features
- **Slow Heart Rate**: RR intervals >1 s (heart rate <60 bpm).
- **Normal Waves**: P, QRS, T waves present, normal morphology (unless conduction block).
- **Possible AV Block**: Prolonged PR (>200 ms) or dropped QRS in second/third-degree AV block.
- **Leads**: Lead II for rate and rhythm.

### 5.4 How to Detect
- **Manual Analysis**:
  - Measure RR interval, calculate heart rate (60/RR in seconds).
  - Check for AV block (e.g., dropped QRS).
- **Automated Detection**:
  - **Features**: RR interval length, PR interval.
  - **Algorithms**: QRS detection, interval measurement.
  - **Libraries**: `wfdb` for RR, `neurokit2` for intervals.
  - **ML/DL**: Classify slow rhythms using time-series models.
- **Challenges**: Differentiate benign (e.g., athlete’s heart) from pathological bradycardia.

### 5.5 How to Solve
- **Clinical**:
  - **Treat Cause**: Adjust medications, correct electrolytes.
  - **Symptomatic**: Pacemaker for persistent bradycardia.
  - **Monitor**: Holter or wearable devices.
- **Signal Processing**:
  - Ensure accurate QRS detection for rate calculation.
  - Use ML/DL for context-aware detection (e.g., sleep vs. activity).

### 5.6 Example
Calculate heart rate:
```python
heart_rate = 60 / np.mean(rr_intervals)
if heart_rate < 60:
    print("Bradycardia detected")
```

**Analogy**: Bradycardia is like a slow waltz, with long pauses between beats, long RR intervals).

---

## 6. Tachycardia

### 6.1 What is Tachycardia?
Tachycardia is a fast heart rate (>100 bpm) that may strain the heart. It’s like a band playing a song too fast, tiring out the musicians (heart muscle).

### 6.2 Physiology
- **Cause**: Increased SA node firing (sinus tachycardia), ectopic foci (e.g., SVT, VT), stress, fever, anemia, thyroid disease.
- **Effect**: Reduced ventricular filling time, lowering cardiac output; may cause palpitations or ischemia.
- **Risk Factors**: Stress, heart disease, caffeine, dehydration, drugs.
- **Complications**: Heart failure, ischemia, conversion to VT/VF.

### 6.3 ECG Features
- **Fast Heart Rate**: RR intervals <0.6 s (heart rate >100 bpm).
- **P Waves**: Present in sinus tachycardia, absent or abnormal in SVT/VT.
- **QRS**: Narrow in sinus/SVT, wide in VT.
- **Leads**: Lead II for rate, V1 for QRS morphology.

### 6.4 How to Detect
- **Manual Analysis**:
  - Measure RR interval, check QRS width, and P wave presence.
  - Differentiate sinus (P present) from VT (wide QRS).
- **Automated Detection**:
  - **Features**: RR interval, QRS width, P wave detection.
  - **Algorithms**: QRS detection, morphology analysis.
  - **Libraries**: `biosppy` for rate, `neurokit2` for P waves.
  - **ML/DL**: CNNs for rhythm classification.
- **Challenges**: Distinguishing sinus tachycardia from pathological rhythms.

### 6.5 How to Solve
- **Clinical**:
  - **Treat Cause**: Address triggers (e.g., fever, anxiety).
  - **Medications**: Beta-blockers for sinus tachycardia, antiarrhythmics for SVT.
  - **Emergency**: Cardioversion for unstable tachycardia.
- **Signal Processing**:
  - Enhance P/QRS with filters for accurate detection.
  - Train ML/DL models for real-time monitoring.

### 6.6 Example
Check heart rate:
```python
if heart_rate > 100:
    print("Tachycardia detected")
```

**Analogy**: Tachycardia is like a techno track playing too fast, rushing the beat (short RR intervals).

---

## 7. Premature Ventricular Contractions (PVCs)

### 7.1 What are PVCs?
PVCs are early beats originating from the ventricles, interrupting the normal rhythm. They’re like random drum hits in a steady song.

### 7.2 Physiology
- **Cause**: Ectopic focus in ventricles, triggered by stress, caffeine, heart disease, or electrolytes.
- **Effect**: Single PVCs are usually benign; frequent PVCs (>10/hour) may reduce cardiac output.
- **Risk Factors**: MI, heart failure, hypokalemia, stimulants.
- **Complications**: Frequent PVCs may trigger VT or indicate underlying disease.

### 7.3 ECG Features
- **Wide QRS**: >120 ms, bizarre shape, no preceding P wave.
- **Compensatory Pause**: Longer RR interval after PVC (ventricles reset).
- **T Wave**: Opposite direction to QRS (discordant).
- **Leads**: V1–V6 show abnormal QRS morphology.

### 7.4 How to Detect
- **Manual Analysis**:
  - Look for early, wide QRS with no P wave, followed by a pause.
- **Automated Detection**:
  - **Features**: QRS width, RR irregularity, morphology.
  - **Algorithms**: QRS detection, template matching.
  - **Libraries**: `neurokit2` for QRS, `scipy` for morphology.
  - **ML/DL**: CNNs for beat classification.
- **Challenges**: Differentiating PVCs from aberrantly conducted beats.

### 7.5 How to Solve
- **Clinical**:
  - **Benign**: Monitor, reduce triggers (e.g., caffeine).
  - **Frequent**: Beta-blockers, ablation for symptomatic cases.
  - **Treat Cause**: Correct heart disease or electrolytes.
- **Signal Processing**:
  - Segment beats for morphology analysis.
  - Train ML/DL models for PVC detection in wearables.

### 7.6 Example
Detect PVCs:
```python
for i, (q, s) in enumerate(zip(info['ECG_Q_Peaks'], info['ECG_S_Peaks'])):
    if (s - q) / fs > 0.12 and np.isnan(info['ECG_P_Peaks'][i]):
        print("PVC detected at beat", i)
```

**Analogy**: PVCs are like a drummer sneaking in an extra, loud beat (wide QRS) that throws off the rhythm.

---

## 8. Premature Atrial Contractions (PACs)

### 8.1 What are PACs?
PACs are early beats originating from the atria, disrupting the normal rhythm. They’re like an extra guitar strum before the main beat.

### 8.2 Physiology
- **Cause**: Ectopic focus in atria, triggered by stress, caffeine, or atrial dilation.
- **Effect**: Usually benign, may cause palpitations; frequent PACs can trigger AF.
- **Risk Factors**: Heart disease, thyroid disorders, alcohol.
- **Complications**: Rare, but frequent PACs may indicate atrial pathology.

### 8.3 ECG Features
- **Early P Wave**: Abnormal shape or polarity, earlier than expected.
- **Normal QRS**: Follows P wave, narrow unless aberrant conduction.
- **Non-Compensatory Pause**: Shorter pause than PVCs (atria reset SA node).
- **Leads**: Lead II, V1 for P wave clarity.

### 8.4 How to Detect
- **Manual Analysis**:
  - Identify early P wave with normal QRS, check pause.
- **Automated Detection**:
  - **Features**: P wave timing, RR irregularity, P morphology.
  - **Algorithms**: P wave detection, RR analysis.
  - **Libraries**: `neurokit2` for P waves, `biosppy` for QRS.
  - **ML/DL**: CNNs for beat classification.
- **Challenges**: P waves may be small or hidden in noise.

### 8.5 How to Solve
- **Clinical**:
  - **Benign**: Monitor, reduce triggers.
  - **Frequent**: Beta-blockers, treat underlying cause.
- **Signal Processing**:
  - Enhance P waves with filters (5–40 Hz).
  - Use ML/DL for PAC detection in long-term monitoring.

### 8.6 Example
Detect PACs:
```python
for i, p in enumerate(info['ECG_P_Peaks']):
    if i > 0 and (r_peaks[i] - p) / fs < 0.1:  # Early P
        print("PAC detected at beat", i)
```

**Analogy**: PACs are like a guitarist playing an extra note (early P wave) before the drumbeat (QRS).

---

# End-to-End Example: Analyzing ECG for Arrhythmias

Imagine we’re a PhD student tasked with analyzing an ECG from the **MIT-BIH Arrhythmia Database** to detect arrhythmias for an ML/DL project. we’ll preprocess the signal, extract features (e.g., RR intervals, QRS width), apply diagnostic rules to identify arrhythmias, train a convolutional neural network (CNN) to classify beats, and visualize the results. This example simulates a real-world workflow for ECG analysis, preparing we for research in biomedical signal processing.

## Step 1: Load Data (WFDB)
We’ll use the `wfdb` library to load an ECG record from the MIT-BIH Arrhythmia Database, which contains annotated ECGs with labels for normal beats (‘N’) and arrhythmias (e.g., ‘V’ for PVC, ‘A’ for PAC).

```python
import wfdb
import numpy as np

# Load ECG record (10 seconds, 360 Hz)
record = wfdb.rdrecord('mit-bih/100', sampto=10000)
signal = record.p_signal[:, 0]  # Lead I
fs = record.fs  # Sampling rate (360 Hz)
annotation = wfdb.rdann('mit-bih/100', 'atr', sampto=10000)
ann_indices = annotation.sample  # Annotation locations
ann_labels = annotation.symbol  # Labels (e.g., 'N', 'V', 'A')
```

**What’s Happening**: The ECG signal is a time-series of voltage values (mV) representing heart activity. Annotations mark QRS peaks and label beats (e.g., ‘V’ for PVC). We’re loading 10 seconds (10000 samples at 360 Hz) from Lead I.

**Analogy**: This is like opening a music file (ECG) with sheet music (annotations) highlighting key notes (QRS peaks) and labeling special chords (arrhythmias).

## Step 2: Preprocess Signal (SciPy)
Preprocessing cleans the ECG to remove noise (e.g., baseline wander, power line interference) and enhance features like P, QRS, and T waves. We’ll apply a band-pass filter (0.5–40 Hz) and a notch filter (60 Hz).

```python
from scipy.signal import butter, filtfilt, iirnotch

def preprocess_ecg(data, fs):
    # Band-pass filter (0.5–40 Hz)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40 / nyq
    b, a = butter(4, [low, high], btype='band')
    data_filt = filtfilt(b, a, data)
    # Notch filter (60 Hz)
    b, a = iirnotch(60, 30, fs)
    return filtfilt(b, a, data_filt)

signal_clean = preprocess_ecg(signal, fs)
```

**What’s Happening**: The band-pass filter keeps frequencies relevant to ECG waves (QRS: 5–15 Hz, P/T: 0.5–10 Hz) while removing low-frequency baseline wander (<0.5 Hz) and high-frequency muscle noise (>40 Hz). The notch filter eliminates 60 Hz power line interference (common in the US).

**Analogy**: This is like tuning a radio to filter out static (noise) and keep the music (ECG waves) clear.

## Step 3: Fiducial Point Detection (NeuroKit2)
We’ll use `neurokit2` to detect key ECG points (P, Q, R, S, T waves) for feature extraction. These points help measure intervals (e.g., RR, PR) and morphology (e.g., QRS width).

```python
import neurokit2 as nk

# Process ECG to detect fiducial points
ecg_signals, info = nk.ecg_process(signal_clean, sampling_rate=fs)
p_peaks = info['ECG_P_Peaks']  # P wave peaks
r_peaks = info['ECG_R_Peaks']  # R wave peaks
q_peaks = info['ECG_Q_Peaks']  # Q wave starts
s_peaks = info['ECG_S_Peaks']  # S wave ends
t_peaks = info['ECG_T_Peaks']  # T wave peaks
```

**What’s Happening**: `neurokit2` applies algorithms (e.g., Pan-Tompkins for QRS) to locate waves. R peaks are the tallest, most reliable points, used as anchors to find P (before R) and T (after S). Q and S define QRS boundaries.

**Analogy**: This is like marking the main beats (R peaks) in a song, then finding the intro (P) and outro (T) notes around them.

## Step 4: Feature Extraction (NumPy)
We’ll extract features for each beat to identify arrhythmias:
- **RR Intervals**: Time between R peaks (heart rate, regularity).
- **QRS Width**: Duration of QRS complex (normal vs. wide).
- **P Wave Presence**: Presence/absence and timing of P waves.
- **Entropy**: Signal complexity (high for VF, moderate for AF).
- **Frequency Power**: Power in 4–10 Hz band (for AF, flutter).

```python
import numpy as np
from scipy.signal import welch

# RR intervals and heart rate
rr_intervals = np.diff(r_peaks) / fs  # Seconds
heart_rate = 60 / np.mean(rr_intervals)  # BPM
rr_std = np.std(rr_intervals)  # Variability

# QRS widths
qrs_widths = [(s - q) / fs for q, s in zip(q_peaks, s_peaks) if not (np.isnan(q) or np.isnan(s))]

# P wave presence
p_presence = len(p_peaks) / len(r_peaks) if len(r_peaks) > 0 else 0

# Entropy
entropy = nk.entropy_approximate(signal_clean)

# Frequency analysis (4–10 Hz for AF/flutter)
freqs, psd = welch(signal_clean, fs=fs, nperseg=1024)
f_wave_power = np.sum(psd[(freqs >= 4) & (freqs <= 10)])

print(f"Heart Rate: {heart_rate:.1f} BPM, RR Std: {rr_std:.3f} s, QRS Width: {np.mean(qrs_widths):.3f} s")
print(f"P Presence: {p_presence:.2f}, Entropy: {entropy:.2f}, F-Wave Power: {f_wave_power:.2e}")
```

**What’s Happening**: These features capture rhythm (RR, heart rate), morphology (QRS width, P presence), and signal complexity (entropy, frequency power). For example, AF has high RR variability and no P waves, while VT has wide QRS.

**Analogy**: This is like analyzing a song’s tempo (RR), chord shapes (QRS), and background hum (f-waves) to identify its genre (arrhythmia).

## Step 5: Diagnostic Rules
We’ll apply rule-based logic to detect arrhythmias based on ECG features, mimicking clinical guidelines. These rules serve as a baseline before ML/DL.

```python
diagnoses = []

# Bradycardia
if heart_rate < 60:
    diagnoses.append('Bradycardia')

# Tachycardia
if heart_rate > 100:
    diagnoses.append('Tachycardia')

# Atrial Fibrillation
if rr_std > 0.1 and p_presence < 0.5:
    diagnoses.append('Atrial Fibrillation')

# Atrial Flutter
if f_wave_power > np.percentile(psd, 95) and 4 <= freqs[np.argmax(psd)] <= 6:
    diagnoses.append('Atrial Flutter')

# Ventricular Tachycardia
if np.mean(qrs_widths) > 0.12 and heart_rate > 100:
    diagnoses.append('Ventricular Tachycardia')

# Ventricular Fibrillation
if entropy > 1.5 and len(r_peaks) < 5:  # Few or no QRS
    diagnoses.append('Ventricular Fibrillation')

# PVC and PAC
beat_labels = []
for i, (q, s, p, r) in enumerate(zip(q_peaks, s_peaks, p_peaks, r_peaks)):
    qrs_width = (s - q) / fs if not (np.isnan(q) or np.isnan(s)) else 0
    pr = (r - p) / fs if not np.isnan(p) else float('inf')
    if qrs_width > 0.12 and np.isnan(p):
        beat_labels.append(f'PVC at beat {i}')
    elif i > 0 and pr < 0.1:  # Early P
        beat_labels.append(f'PAC at beat {i}')
    else:
        beat_labels.append('Normal')
diagnoses.extend(beat_labels[:5])  # Limit to 5 for brevity

print(f"Diagnoses: {diagnoses if diagnoses else 'No clear abnormalities'}")
```

**What’s Happening**: Rules check heart rate thresholds (bradycardia, tachycardia), RR irregularity and P waves (AF), frequency power (flutter), QRS width and rate (VT), entropy (VF), and beat-specific anomalies (PVC, PAC). These are simplified but align with clinical criteria (e.g., AHA/ESC).

**Analogy**: This is like a music teacher checking if the song’s too slow (bradycardia), has no intro notes (AF, no P waves), or includes wild drum solos (PVCs).

## Step 6: Deep Learning Model (TensorFlow)
To go beyond rules, we’ll train a CNN to classify beats as normal (‘N’), PVC (‘V’), or PAC (‘A’) using segmented ECGs. This simulates a real ML/DL research task.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Segment ECG around R peaks (300 ms window, ~108 samples at 360 Hz)
segments = []
labels = []
window_size = 108  # ~150 ms before/after R peak
for i, r in enumerate(r_peaks):
    if r >= window_size//2 and r < len(signal_clean) - window_size//2:
        segment = signal_clean[r - window_size//2:r + window_size//2]
        if len(segment) == window_size:
            segments.append(segment)
            # Map annotations to labels
            ann_idx = np.argmin(np.abs(ann_indices - r))
            if ann_idx < len(ann_labels):
                if ann_labels[ann_idx] == 'N':
                    labels.append(0)  # Normal
                elif ann_labels[ann_idx] == 'V':
                    labels.append(1)  # PVC
                elif ann_labels[ann_idx] == 'A':
                    labels.append(2)  # PAC

# Prepare data
X = np.array(segments)[:, :, np.newaxis]  # Shape: (n_segments, window_size, 1)
y = tf.keras.utils.to_categorical(labels)  # One-hot encode

# Build CNN
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size, 1)),
    Conv1D(64, 5, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Normal, PVC, PAC
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Predict on first 5 segments
predictions = model.predict(X[:5])
pred_labels = ['Normal', 'PVC', 'PAC']
print("Predictions:", [pred_labels[np.argmax(p)] for p in predictions])
```

**What’s Happening**: We segment the ECG around R peaks (300 ms windows), label segments using annotations, and train a CNN to classify beats. The CNN learns patterns like wide QRS (PVC) or early P waves (PAC). In practice, we’d use more data (e.g., entire MIT-BIH dataset) and balance classes.

**Analogy**: The CNN is like a music critic learning to spot normal melodies, rogue drum hits (PVCs), or extra guitar strums (PACs) by studying the score (ECG segments).

## Step 7: Visualize Results (Matplotlib)
Let’s plot the ECG with detected R peaks, P waves, and beat labels to inspect our findings.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(signal_clean, label='Filtered ECG', alpha=0.7)
plt.plot(r_peaks, signal_clean[r_peaks], 'ro', label='R Peaks')
plt.plot(p_peaks, signal_clean[p_peaks], 'go', label='P Peaks', alpha=0.5)
# Annotate beats
for i, r in enumerate(r_peaks[:10]):  # Limit for clarity
    if i < len(beat_labels):
        plt.text(r, signal_clean[r] + 0.1, beat_labels[i], fontsize=8)
plt.title('ECG with Detected Arrhythmias')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.show()
```

**What’s Happening**: The plot shows the cleaned ECG with R peaks (red), P peaks (green), and text labels for detected arrhythmias (e.g., ‘PVC’, ‘Normal’). This helps visualize rule-based and ML/DL results.

**Analogy**: This is like marking up sheet music with notes (R peaks) and comments (arrhythmia labels) to show where the band went off-key.

## Step 8: Summarize
- **Findings**: We loaded an ECG, preprocessed it to remove noise, detected fiducial points (P, QRS, T), extracted features (RR, QRS width, entropy), applied diagnostic rules to identify arrhythmias (e.g., AF, PVC), and trained a CNN to classify beats. The visualization confirmed detected beats and potential arrhythmias.
- **Outcome**: The pipeline produces features and classifications ready for ML/DL research, suitable for real-time monitoring or large-scale studies.
- **Next Steps**:
  - Expand to multi-lead ECGs (e.g., 12-lead PTB-XL dataset).
  - Balance classes (e.g., oversample PVCs/PACs) for better CNN performance.
  - Implement real-time detection for wearables using optimized models.
  - Explore advanced DL (e.g., LSTMs for rhythm sequences, transformers for multi-lead).

## Tips for PhD Preparation
- **Practice**: Download the MIT-BIH Arrhythmia Database from PhysioNet and replicate this example. Try other records (e.g., 108 for AF, 200 for PVCs).
- **Visualize**: Plot ECGs with annotations to understand patterns (e.g., irregular RR in AF, wide QRS in VT).
- **Analogies**: Recall arrhythmias as music genres—jazz (AF), carousel (flutter), heavy metal (VT), static (VF), waltz (bradycardia), techno (tachycardia), rogue drum (PVC), extra strum (PAC).
- **ML/DL Focus**:
  - Use datasets like PTB-XL or CinC Challenge for diverse arrhythmias.
  - Experiment with feature-based ML (e.g., SVM with RR, entropy) vs. end-to-end DL (e.g., CNNs on raw ECG).
  - Study papers on ECG classification (e.g., PhysioNet/Computing in Cardiology).
- **Tools**: Master `wfdb`, `neurokit2`, `biosppy`, `scipy`, `tensorflow`, and `pytorch`. Explore `pywavelets` for advanced denoising.
- **Research Ideas**:
  - Real-time AF detection in wearables.
  - Differentiating VT from SVT using multi-lead morphology.
  - Predicting VF onset using HRV and TWA features.
