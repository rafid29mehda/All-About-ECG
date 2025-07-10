# ECG Wave Analysis Techniques 

## 1. QRS Complex Detection (Pan-Tompkins Algorithm)

### What is QRS Complex Detection?
The **QRS complex** is the tall, sharp spike in an ECG signal (Q dip, R peak, S dip) that represents the ventricles contracting to pump blood. Detecting QRS complexes is like finding the “beat” in the heart’s music, as it’s the most prominent feature and helps calculate heart rate.

### What is the Pan-Tompkins Algorithm?
The **Pan-Tompkins algorithm** is a widely used method to detect QRS complexes in ECG signals. It’s like a detective that searches for the tallest peaks (R waves) while ignoring noise and other waves (P, T).

### How It Works (Step-by-Step)
1. **Band-Pass Filtering** (5–15 Hz):
   - Removes noise (e.g., baseline wander <1 Hz, muscle artifacts >40 Hz) and keeps QRS frequencies.
   - Why? QRS complexes have energy mostly in the 5–15 Hz range.
2. **Differentiation**:
   - Calculates the signal’s slope to highlight sharp changes (QRS has steep slopes).
3. **Squaring**:
   - Squares the signal to amplify peaks and make all values positive, enhancing QRS.
4. **Moving Window Integration**:
   - Smooths the signal by summing values in a window (e.g., 150 ms) to create peaks at QRS locations.
5. **Thresholding**:
   - Sets a threshold to identify peaks as QRS complexes.
   - Uses adaptive thresholds to adjust for signal changes.
6. **Decision Rules**:
   - Checks timing (e.g., QRS complexes are ~0.06–0.12 seconds apart) to avoid false positives.

### Why It’s Important for ECG Research
- QRS detection is the foundation for heart rate calculation and rhythm analysis.
- Accurate QRS locations are critical for ML/DL models (e.g., classifying arrhythmias).
- The Pan-Tompkins algorithm is robust, real-time capable, and widely implemented.

**Example**: Imagine an ECG as a bumpy road. The Pan-Tompkins algorithm is like a GPS that finds the tallest hills (R peaks) while ignoring small bumps (noise or P/T waves).

---

## 2. Heart Rate Variability (HRV) Metrics

### What is HRV?
**Heart rate variability (HRV)** measures the variation in time between consecutive heartbeats (RR intervals, the time between R peaks). It’s like checking how steady or flexible the heart’s rhythm is, reflecting the autonomic nervous system’s control.

### Why It Matters
- High HRV: Healthy, adaptable heart (e.g., responds well to stress).
- Low HRV: May indicate stress, fatigue, or heart disease.
- Used in research for studying stress, sleep, or conditions like heart failure.

### Common HRV Metrics
1. **Time-Domain Metrics**:
   - **SDNN**: Standard deviation of RR intervals (overall variability, ms).
   - **RMSSD**: Root mean square of successive RR differences (short-term variability, ms).
   - **pNN50**: Percentage of RR intervals differing by >50 ms (parasympathetic activity).
2. **Frequency-Domain Metrics**:
   - **LF (Low Frequency, 0.04–0.15 Hz)**: Reflects sympathetic and parasympathetic activity.
   - **HF (High Frequency, 0.15–0.4 Hz)**: Reflects parasympathetic activity (breathing-related).
   - **LF/HF Ratio**: Balance between sympathetic and parasympathetic systems.
3. **Nonlinear Metrics**:
   - **Poincaré Plot**: Scatter plot of RR(n) vs. RR(n+1) to visualize variability.
   - **Entropy**: Measures signal complexity (e.g., approximate entropy).

### How to Calculate HRV
- Detect R peaks (e.g., using Pan-Tompkins).
- Calculate RR intervals (time between R peaks in seconds).
- Apply metrics using libraries like `neurokit2` or `hrv-analysis`.

### Why It’s Important for ECG Research
- HRV metrics are features for ML/DL models to predict stress, disease, or fitness.
- Noninvasive and rich in physiological insights, ideal for PhD research.

**Example**: Think of HRV as the rhythm of a drummer (heart). A good drummer varies the beat slightly (high HRV), while a robotic drummer is too rigid (low HRV).

---

## 3. QT Interval Measurement

### What is the QT Interval?
The **QT interval** is the time from the start of the Q wave to the end of the T wave in an ECG. It represents the ventricles’ electrical activity (contraction and relaxation). It’s like measuring how long the heart’s “pump and rest” cycle takes.

### Why It Matters
- **Normal QT**: 0.35–0.44 seconds (varies with heart rate).
- **Long QT**: >0.44 seconds, risks arrhythmias like Torsades de Pointes.
- **Short QT**: <0.35 seconds, rare but also risky.
- Used to assess drug effects, electrolyte imbalances, or genetic conditions.

### How to Measure
1. **Identify Q and T Points**:
   - Q: Start of QRS (first downward deflection).
   - T: End of T wave (return to baseline).
   - Use algorithms or manual inspection (T end is tricky due to gradual slope).
2. **Calculate Duration**:
   - Measure time (in seconds) from Q start to T end.
   - Use sampling rate (e.g., at 250 Hz, 100 samples = 0.4 seconds).
3. **Correct for Heart Rate (QTc)**:
   - Heart rate affects QT, so correct using Bazett’s formula:
     \[
     QTc = \frac{QT}{\sqrt{RR}}
     \]
     where RR is the RR interval in seconds.
   - Normal QTc: 0.36–0.44 seconds (men), 0.36–0.46 seconds (women).

### Challenges
- T wave end is hard to detect (fuzzy boundary).
- Noise or baseline wander can distort measurements.
- Automated algorithms (e.g., in `neurokit2`) improve consistency.

### Why It’s Important for ECG Research
- QT measurement is critical for drug safety studies and arrhythmia risk assessment.
- QT features can be used in ML/DL for classifying cardiac conditions.

**Example**: Imagine the QT interval as the time a runner (ventricles) takes to sprint (contract) and walk back (relax). Measuring it tells you if they’re too slow or fast.

---

## 4. ST Segment Analysis

### What is the ST Segment?
The **ST segment** is the flat part of the ECG between the S wave and the T wave, representing the ventricles’ “pause” between contraction and relaxation. It’s like the calm moment after the heart’s big pump.

### Why It Matters
- **Normal ST**: Level with the baseline (isoelectric line).
- **ST Elevation**: Raised above baseline (>1 mm), often indicates a heart attack (myocardial infarction).
- **ST Depression**: Below baseline, suggests ischemia (reduced blood flow).
- Used for diagnosing acute cardiac events.

### How to Analyze
1. **Identify ST Segment**:
   - Start: J point (end of S wave).
   - End: Start of T wave.
2. **Measure Relative to Baseline**:
   - Baseline: Flat line between T and P waves (isoelectric line).
   - Measure ST elevation/depression in mm (1 mm = 0.1 mV) at a fixed point (e.g., 60 ms after J point).
3. **Interpret by Lead**:
   - Elevation in II, III, aVF: Inferior heart attack.
   - Elevation in V1–V4: Anterior heart attack.
4. **Automated Analysis**:
   - Use algorithms to detect deviations (e.g., `neurokit2` or custom thresholding).

### Challenges
- Baseline wander or noise can mimic ST changes.
- Subtle changes (<1 mm) require high-precision ECGs.
- Multiple leads must be analyzed for localization.

### Why It’s Important for ECG Research
- ST segment is a key feature for ML/DL models detecting heart attacks or ischemia.
- Automated ST analysis is a focus for real-time monitoring systems.

**Example**: Think of the ST segment as a tightrope walker. If they’re level (normal), all’s well. If they’re too high (elevation) or too low (depression), it’s a sign of heart trouble.

---

## 5. T Wave Alternans Detection

### What is T Wave Alternans?
**T wave alternans (TWA)** is a subtle, beat-to-beat variation in the T wave’s amplitude or shape, often invisible to the eye. It’s like a tiny flicker in the heart’s relaxation phase, signaling electrical instability.

### Why It Matters
- TWA is a risk marker for sudden cardiac death or ventricular arrhythmias.
- Often seen in patients with heart disease or during stress tests.
- Measured in microvolts (µV), requiring sensitive detection.

### How to Detect
1. **Preprocess Signal**:
   - Remove noise (e.g., baseline wander, muscle artifacts) using filters.
   - Align ECG beats by R peaks for consistency.
2. **Extract T Waves**:
   - Identify T wave boundaries (after QRS, before next P).
3. **Measure Alternans**:
   - Compare T wave amplitudes between consecutive beats.
   - Use spectral methods (e.g., Fourier transform) to detect periodic alternans at 0.5 cycles/beat.
4. **Quantify**:
   - TWA amplitude >1.9 µV is significant.
   - Use signal averaging to enhance small alternans.
5. **Tools**:
   - Libraries like `neurokit2` or custom algorithms for T wave analysis.
   - Specialized software for clinical TWA detection.

### Challenges
- TWA is tiny (µV), easily masked by noise.
- Requires high-quality, high-resolution ECGs (e.g., 1000 Hz, 16-bit).
- False positives from artifacts or irregular rhythms.

### Why It’s Important for ECG Research
- TWA is a niche but powerful feature for ML/DL models predicting arrhythmic risk.
- Research into TWA could lead to better risk stratification tools.

**Example**: Imagine T waves as ripples in a pond after each heartbeat. T wave alternans is like every other ripple being slightly bigger or smaller, hinting at an unstable heart.

---

## End-to-End Example: Analyzing an ECG Signal

Let’s imagine you’re a PhD student analyzing an ECG from the MIT-BIH Arrhythmia Database to extract features for an ML model to detect arrhythmias. You’ll use Python and libraries like `wfdb`, `biosppy`, `neurokit2`, and `scipy`.

### Step 1: Load Data (WFDB)
- Load ECG and annotations:
  ```python
  import wfdb
  record = wfdb.rdrecord('mit-bih/100', sampto=10000)
  signal = record.p_signal[:, 0]  # Lead I
  fs = record.fs  # 360 Hz
  annotation = wfdb.rdann('mit-bih/100', 'atr', sampto=10000)
  qrs_indices = annotation.sample
  labels = annotation.symbol  # e.g., 'N' (normal), 'V' (PVC)
  ```

### Step 2: QRS Detection (Pan-Tompkins via BioSPPy)
- Apply Pan-Tompkins algorithm:
  ```python
  from biosppy.signals import ecg
  ecg_out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
  r_peaks = ecg_out['rpeaks']  # QRS R peaks
  ```
- Verify against annotations:
  ```python
  print(f"Detected {len(r_peaks)} QRS peaks, {len(qrs_indices)} annotated")
  ```

### Step 3: HRV Analysis (NeuroKit2)
- Calculate RR intervals and HRV metrics:
  ```python
  import neurokit2 as nk
  ecg_signals, info = nk.ecg_process(signal, sampling_rate=fs)
  r_peaks = info['ECG_R_Peaks']
  rr_intervals = np.diff(r_peaks) / fs  # Seconds
  hrv_metrics = nk.hrv_time(r_peaks, sampling_rate=fs)
  print(f"SDNN: {hrv_metrics['HRV_SDNN'].iloc[0]:.2f} ms, RMSSD: {hrv_metrics['HRV_RMSSD'].iloc[0]:.2f} ms")
  ```

### Step 4: QT Interval Measurement (NeuroKit2)
- Measure QT intervals:
  ```python
  q_peaks = info['ECG_Q_Peaks']
  t_peaks = info['ECG_T_Peaks']
  qt_intervals = (t_peaks - q_peaks) / fs  # Seconds
  rr_mean = np.mean(rr_intervals)
  qtc = np.mean(qt_intervals) / np.sqrt(rr_mean)  # Bazett’s formula
  print(f"QTc: {qtc:.3f} seconds")
  ```

### Step 5: ST Segment Analysis (Custom)
- Analyze ST segment:
  ```python
  import numpy as np
  st_values = []
  for r in r_peaks:
      j_point = r + int(0.04 * fs)  # 40 ms after R (J point)
      st_point = j_point + int(0.06 * fs)  # 60 ms after J
      if st_point < len(signal):
          baseline = np.mean(signal[r-50:r-10])  # Estimate baseline
          st_deviation = signal[st_point] - baseline  # mV
          st_values.append(st_deviation)
  print(f"Average ST deviation: {np.mean(st_values):.3f} mV")
  ```

### Step 6: T Wave Alternans Detection (Custom)
- Detect TWA (simplified):
  ```python
  t_amplitudes = []
  for t in t_peaks:
      if t-50 < len(signal) and t+50 < len(signal):
          t_amplitude = np.max(signal[t-50:t+50]) - np.min(signal[t-50:t+50])
          t_amplitudes.append(t_amplitude)
  alternans = np.abs(np.diff(t_amplitudes[::2]))  # Compare every other T wave
  print(f"Average TWA amplitude: {np.mean(alternans)*1000:.2f} µV")
  ```

### Step 7: Visualize (Matplotlib)
- Plot results:
  ```python
  import matplotlib.pyplot as plt
  plt.plot(signal, label='ECG')
  plt.plot(r_peaks, signal[r_peaks], 'ro', label='R Peaks')
  plt.plot(q_peaks, signal[q_peaks], 'go', label='Q Peaks')
  plt.plot(t_peaks, signal[t_peaks], 'bo', label='T Peaks')
  plt.title('ECG Analysis')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude (mV)')
  plt.legend()
  plt.show()
  ```

### Step 8: Summarize
- **Findings**: The ECG was analyzed for QRS complexes (Pan-Tompkins), HRV (SDNN, RMSSD), QTc (Bazett’s), ST deviation, and TWA. Features like RR intervals and ST values are ready for ML/DL.
- **Outcome**: The extracted features can train a model to detect arrhythmias or assess cardiac risk.
- **Next Steps**: Use features in a TensorFlow/PyTorch model or analyze more data from PTB-XL.


## 6. P Wave Identification

### What is P Wave Identification?
The **P wave** is the small, rounded bump in an ECG before the QRS complex, representing the atria contracting to push blood into the ventricles. Identifying P waves is like spotting the soft “intro” notes before the loud “beat” (QRS) in the heart’s rhythm.

### Why Matters
- P waves indicate atrial activity, essential for diagnosing atrial arrhythmias (e.g., atrial fibrillation, where P waves may be absent or irregular).
- Their shape, size, and timing help assess atrial health (e.g., enlarged atria).
- Critical for ML/DL models to classify rhythms or detect abnormalities.

### How to Identify P Waves
1. **Preprocess Signal**:
   - Filter ECG to remove noise (e.g., baseline wander <1 Hz, muscle artifacts >40 Hz) using a band-pass filter (0.5–40 Hz).
2. **Locate QRS Complexes**:
   - Use algorithms like Pan-Tompkins to find R peaks (tallest QRS points).
   - P waves occur ~100–200 ms before the R peak.
3. **Search for P Waves**:
   - Look for small positive (or biphasic) waves in the PR interval (region before QRS).
   - Use peak detection with a low threshold in the window 200 ms before R.
4. **Automated Tools**:
   - Libraries like `neurokit2` or `biosppy` detect P waves automatically.
   - Algorithms may use wavelet transforms to isolate P wave shapes.
5. **Manual Verification**:
   - Plot ECG and check if detected P waves align with expected timing and shape.

### Challenges
- P waves are small (0.1–0.25 mV), easily hidden by noise or overlapping T waves.
- Absent or irregular P waves in arrhythmias (e.g., atrial fibrillation) complicate detection.
- Lead selection matters (e.g., Lead II often shows clear P waves).

### Why It’s Important for ECG Research
- P wave features (e.g., duration, amplitude) are inputs for ML/DL models to detect atrial issues.
- Accurate P wave detection improves rhythm classification accuracy.

**Example**: Imagine an ECG as a song. The P wave is the quiet guitar strum before the loud drumbeat (QRS). Identifying it helps you understand the song’s structure.

---

## 7. Rhythm Classification (Sinus, Atrial, Ventricular)

### What is Rhythm Classification?
**Rhythm classification** involves identifying the heart’s electrical rhythm based on ECG patterns. It’s like determining the genre of the heart’s music—normal (sinus), atrial (e.g., fibrillation), or ventricular (e.g., tachycardia).

### Common Rhythms
1. **Normal Sinus Rhythm (NSR)**:
   - Regular rhythm, heart rate 60–100 beats/min.
   - Clear P wave before each QRS, consistent PR interval (~120–200 ms).
   - QRS narrow (<120 ms).
2. **Atrial Rhythms**:
   - **Atrial Fibrillation (AF)**: Irregular rhythm, no clear P waves, wavy baseline (fibrillatory waves), irregular QRS spacing.
   - **Atrial Flutter**: Sawtooth-like flutter waves (2:1 or 4:1 ratio with QRS), regular or irregular QRS.
   - **Premature Atrial Contraction (PAC)**: Early beat with abnormal P wave, followed by QRS.
3. **Ventricular Rhythms**:
   - **Ventricular Tachycardia (VT)**: Fast, wide QRS complexes (>120 ms), no clear P waves, regular rhythm.
   - **Ventricular Fibrillation (VF)**: Chaotic, irregular waves, no recognizable QRS or P waves.
   - **Premature Ventricular Contraction (PVC)**: Early, wide QRS with no preceding P wave.

### How to Classify
1. **Feature Extraction**:
   - Measure RR intervals (time between R peaks), P wave presence, QRS width, and PR intervals.
   - Analyze regularity (e.g., irregular RR in AF).
2. **Rule-Based Classification**:
   - Example: If P wave present, PR 120–200 ms, QRS <120 ms, and RR regular, classify as NSR.
   - Use guidelines like AHA/ACC for criteria.
3. **Automated Methods**:
   - ML models (e.g., SVM, Random Forest) use features like RR variability, QRS morphology.
   - DL models (e.g., CNNs, LSTMs) learn patterns directly from raw ECGs.
   - Libraries like `neurokit2` provide rhythm analysis tools.
4. **Lead Selection**:
   - Lead II for P waves, V1 for QRS morphology, multiple leads for comprehensive analysis.

### Challenges
- Noise or artifacts can mimic abnormal rhythms.
- Overlapping rhythms (e.g., AF with PVCs) require advanced analysis.
- Requires multiple leads for accurate classification (e.g., VT vs. SVT).

### Why It’s Important for ECG Research
- Rhythm classification is central to diagnosing cardiac conditions.
- ML/DL models for automated rhythm detection are a hot research area, improving real-time monitoring.

**Example**: Think of rhythm classification as identifying a song’s genre. Sinus rhythm is like pop (steady, predictable), atrial fibrillation is like jazz (irregular, chaotic), and ventricular tachycardia is like heavy metal (fast, intense).

---

## 8. Fiducial Point Detection

### What is Fiducial Point Detection?
**Fiducial points** are specific landmarks in the ECG waveform, like the start/end of P, QRS, and T waves. Detecting them is like marking the key moments in a song’s melody (e.g., verse, chorus).

### Key Fiducial Points
- **P Wave**: Start, peak, end.
- **QRS Complex**: Q start, R peak, S end, J point (end of QRS).
- **T Wave**: Start, peak, end.
- **Intervals**: PR (P start to QRS start), QT (Q start to T end).

### Why It Matters
- Fiducial points define intervals and segments (e.g., PR, QT, ST) for diagnosing conditions.
- Essential for feature extraction in ML/DL (e.g., QRS width for ventricular rhythms).
- Used in automated analysis and HRV metrics.

### How to Detect
1. **Preprocess Signal**:
   - Filter ECG (0.5–40 Hz) to remove noise.
2. **QRS Detection**:
   - Use Pan-Tompkins or libraries like `biosppy` to find R peaks as reference points.
3. **P and T Wave Detection**:
   - Search for P waves before R peaks (100–200 ms window).
   - Search for T waves after S waves (200–400 ms window).
   - Use peak detection or wavelet transforms to identify shapes.
4. **Boundary Detection**:
   - Find start/end points by detecting slope changes or zero crossings.
   - Libraries like `neurokit2` automate this.
5. **Validation**:
   - Check timing (e.g., PR <200 ms, QRS <120 ms) to avoid false detections.

### Challenges
- Small waves (P, T) are hard to detect in noisy signals.
- Abnormal rhythms (e.g., AF) may lack clear fiducial points.
- Requires high-resolution ECGs (e.g., 250–500 Hz).

### Why It’s Important for ECG Research
- Fiducial points are critical inputs for ML/DL models analyzing intervals or morphology.
- Automated detection reduces manual effort in large-scale studies.

**Example**: Imagine an ECG as a comic strip. Fiducial points are like the panel borders marking key actions (P, QRS, T), helping you follow the story.

---

## 9. Beat-to-Beat Analysis

### What is Beat-to-Beat Analysis?
**Beat-to-beat analysis** examines each heartbeat (QRS complex and surrounding waves) individually to study variations. It’s like analyzing each note in a song to see how it differs from the last.

### Why It Matters
- Detects irregularities (e.g., PVCs, PACs) that disrupt normal rhythm.
- Provides data for HRV (heart rate variability) and other metrics.
- Identifies transient abnormalities (e.g., intermittent arrhythmias).

### How to Perform
1. **Segment Beats**:
   - Use R peaks (from QRS detection) to segment ECG into beats (e.g., 200 ms before to 400 ms after R).
2. **Extract Features**:
   - **Timing**: RR intervals, PR interval, QRS duration.
   - **Morphology**: QRS shape, P/T wave amplitude, ST deviation.
   - **Annotations**: Label beats as normal, PVC, PAC, etc.
3. **Analyze Variations**:
   - Calculate RR variability for HRV.
   - Compare QRS shapes to detect abnormal beats (e.g., wide QRS for PVC).
   - Use statistical methods (e.g., variance, entropy).
4. **Automated Tools**:
   - Libraries like `neurokit2` or `biosppy` segment and analyze beats.
   - ML/DL models classify beats based on features or raw segments.

### Challenges
- Noise or artifacts can distort beat morphology.
- Irregular rhythms (e.g., AF) complicate segmentation.
- Requires consistent QRS detection for accurate analysis.

### Why It’s Important for ECG Research
- Beat-to-beat features are key inputs for ML/DL models detecting arrhythmias or predicting cardiac events.
- Enables real-time monitoring for wearable devices.

**Example**: Think of beat-to-beat analysis as checking each dance move in a routine. Most moves are smooth (normal beats), but a sudden jump (PVC) stands out.

---

## 10. Automated Diagnostic Rules

### What are Automated Diagnostic Rules?
**Automated diagnostic rules** are algorithms that use ECG features to diagnose conditions without human intervention. They’re like a checklist a computer follows to spot heart problems.

### Why It Matters
- Speeds up diagnosis in clinical settings or wearable devices.
- Reduces human error and subjectivity.
- Provides consistent results for large-scale research or ML/DL validation.

### Common Diagnostic Rules
1. **Heart Rate**:
   - Normal: 60–100 bpm.
   - Bradycardia: <60 bpm.
   - Tachycardia: >100 bpm.
   - Rule: Calculate heart rate from RR intervals (60/RR in seconds).
2. **Arrhythmias**:
   - **Atrial Fibrillation**: Irregular RR intervals, no P waves, fibrillatory baseline.
   - **PVC**: Wide QRS (>120 ms), no preceding P wave.
   - **VT**: ≥3 wide QRS beats at >100 bpm.
   - Rule: Check RR regularity, QRS width, P wave presence.
3. **Conduction Abnormalities**:
   - **First-Degree AV Block**: PR interval >200 ms.
   - **Bundle Branch Block**: QRS >120 ms, specific QRS morphology (e.g., RSR’ in V1 for RBBB).
   - Rule: Measure PR and QRS durations.
4. **Ischemia/Infarction**:
   - **ST Elevation**: >1 mm in ≥2 contiguous leads (e.g., II, III, aVF for inferior MI).
   - **Q Waves**: Pathological Q waves (>40 ms or >25% of R wave) indicate past infarction.
   - Rule: Measure ST deviation and Q wave parameters.
5. **QT Prolongation**:
   - QTc >440 ms (men) or >460 ms (women) using Bazett’s formula.
   - Rule: Measure QT and correct for RR.

### How to Implement
1. **Feature Extraction**:
   - Detect fiducial points (P, QRS, T) and measure intervals (PR, QRS, QT, ST).
   - Use libraries like `neurokit2` or `biosppy`.
2. **Rule-Based Logic**:
   - Write if-then statements (e.g., if QRS >120 ms and no P wave, classify as PVC).
   - Use guidelines from AHA/ACC or ESC.
3. **Automated Systems**:
   - ML models (e.g., decision trees) encode rules implicitly.
   - DL models (e.g., CNNs) learn rules from raw ECGs.
4. **Validation**:
   - Compare against annotated datasets (e.g., MIT-BIH, PTB-XL).
   - Ensure sensitivity/specificity for clinical use.

### Challenges
- Noise or artifacts can trigger false positives.
- Complex conditions (e.g., mixed arrhythmias) may not fit simple rules.
- Rules must be updated with new clinical guidelines.

### Why It’s Important for ECG Research
- Automated rules are the backbone of real-time ECG monitors and wearable devices.
- They provide benchmarks for ML/DL models and enable large-scale diagnostic studies.

**Example**: Imagine automated rules as a robot doctor with a checklist. It scans the ECG, checks boxes (e.g., “P wave present?”, “QRS wide?”), and announces the diagnosis.

---

## End-to-End Example: Analyzing an ECG Signal

Let’s imagine you’re a PhD student analyzing an ECG from the MIT-BIH Arrhythmia Database to extract features and apply diagnostic rules for an ML project on arrhythmia detection. You’ll use Python with libraries like `wfdb`, `biosppy`, `neurokit2`, and `numpy`.

### Step 1: Load Data (WFDB)
- Load ECG and annotations:
  ```python
  import wfdb
  record = wfdb.rdrecord('mit-bih/100', sampto=10000)
  signal = record.p_signal[:, 0]  # Lead I
  fs = record.fs  # 360 Hz
  annotation = wfdb.rdann('mit-bih/100', 'atr', sampto=10000)
  qrs_indices = annotation.sample
  labels = annotation.symbol  # e.g., 'N' (normal), 'V' (PVC)
  ```

### Step 2: Preprocess Signal (SciPy)
- Apply band-pass filter (0.5–40 Hz):
  ```python
  from scipy.signal import butter, filtfilt
  def bandpass_filter(data, fs, lowcut=0.5, highcut=40):
      nyq = 0.5 * fs
      low = lowcut / nyq
      high = highcut / nyq
      b, a = butter(4, [low, high], btype='band')
      return filtfilt(b, a, data)
  signal_filtered = bandpass_filter(signal, fs)
  ```

### Step 3: P Wave Identification (NeuroKit2)
- Detect P waves and other fiducial points:
  ```python
  import neurokit2 as nk
  ecg_signals, info = nk.ecg_process(signal_filtered, sampling_rate=fs)
  p_peaks = info['ECG_P_Peaks']
  r_peaks = info['ECG_R_Peaks']
  q_peaks = info['ECG_Q_Peaks']
  s_peaks = info['ECG_S_Peaks']
  t_peaks = info['ECG_T_Peaks']
  ```

### Step 4: Fiducial Point Detection (NeuroKit2)
- Measure intervals:
  ```python
  import numpy as np
  pr_intervals = (q_peaks - p_peaks) / fs  # PR interval in seconds
  qrs_durations = (s_peaks - q_peaks) / fs  # QRS duration
  print(f"Average PR: {np.nanmean(pr_intervals):.3f} s, QRS: {np.nanmean(qrs_durations):.3f} s")
  ```

### Step 5: Rhythm Classification (Custom Rules)
- Classify beats based on features:
  ```python
  beat_labels = []
  rr_intervals = np.diff(r_peaks) / fs  # RR intervals in seconds
  for i, (p, q, r, s) in enumerate(zip(p_peaks, q_peaks, r_peaks, s_peaks)):
      qrs_width = (s - q) / fs
      pr = (q - p) / fs if not np.isnan(p) else np.nan
      rr_var = np.std(rr_intervals[max(0, i-5):i+5]) if i >= 5 else np.nan
      # Rules
      if np.isnan(p) and rr_var > 0.1:
          beat_labels.append('AF')  # Atrial fibrillation (no P, irregular RR)
      elif qrs_width > 0.12 and np.isnan(p):
          beat_labels.append('PVC')  # Wide QRS, no P
      elif pr > 0.2:
          beat_labels.append('AV Block')  # Long PR
      else:
          beat_labels.append('Normal')
  print(f"Beat classifications: {beat_labels[:10]}")
  ```

### Step 6: Beat-to-Beat Analysis (NumPy)
- Segment and analyze beats:
  ```python
  segments = []
  for r in r_peaks:
      if r-100 < len(signal_filtered) and r+200 < len(signal_filtered):
          segment = signal_filtered[r-100:r+200]  # 300 ms window
          segments.append(segment)
  segments = np.array(segments)
  # Calculate feature: max amplitude per beat
  max_amplitudes = np.max(segments, axis=1)
  print(f"Average max amplitude: {np.mean(max_amplitudes):.3f} mV")
  ```

### Step 7: Automated Diagnostic Rules (Custom)
- Apply diagnostic rules:
  ```python
  heart_rate = 60 / np.mean(rr_intervals)
  st_values = []
  for r in r_peaks:
      j_point = r + int(0.04 * fs)  # 40 ms after R
      st_point = j_point + int(0.06 * fs)  # 60 ms after J
      if st_point < len(signal_filtered):
          baseline = np.mean(signal_filtered[r-50:r-10])
          st_deviation = signal_filtered[st_point] - baseline
          st_values.append(st_deviation)
  diagnosis = []
  if heart_rate < 60:
      diagnosis.append('Bradycardia')
  elif heart_rate > 100:
      diagnosis.append('Tachycardia')
  if np.mean(st_values) > 0.1:
      diagnosis.append('ST Elevation')
  if np.nanmean(pr_intervals) > 0.2:
      diagnosis.append('First-Degree AV Block')
  print(f"Diagnosis: {diagnosis}")
  ```

### Step 8: Visualize (Matplotlib)
- Plot results:
  ```python
  import matplotlib.pyplot as plt
  plt.plot(signal_filtered, label='Filtered ECG')
  plt.plot(p_peaks, signal_filtered[p_peaks], 'go', label='P Peaks')
  plt.plot(r_peaks, signal_filtered[r_peaks], 'ro', label='R Peaks')
  plt.plot(t_peaks, signal_filtered[t_peaks], 'bo', label='T Peaks')
  plt.title('ECG Analysis with Fiducial Points')
  plt.xlabel('Sample')
  plt.ylabel('Amplitude (mV)')
  plt.legend()
  plt.show()
  ```

### Step 9: Summarize
- **Findings**: The ECG was preprocessed, P waves identified, fiducial points detected, rhythms classified (e.g., Normal, PVC), beat-to-beat features extracted, and diagnostic rules applied (e.g., heart rate, ST elevation).
- **Outcome**: The extracted features and diagnoses are ready for ML/DL (e.g., training a CNN for arrhythmia detection).
- **Next Steps**: Use features in a TensorFlow/PyTorch model or analyze more data from PTB-XL.
