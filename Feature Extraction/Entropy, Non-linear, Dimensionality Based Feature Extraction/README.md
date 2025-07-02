## 6.7 Entropy-Based Features: Extraction Techniques

Entropy-based features measure the unpredictability or randomness of an ECG signal. Below are the extraction techniques for **Shannon Entropy**, **Approximate Entropy (ApEn)**, **Sample Entropy (SampEn)**, **Multiscale Entropy (MSE)**, **Permutation Entropy**, **Spectral Entropy**, **Wavelet Entropy**, **Renyi Entropy**, **Tsallis Entropy**, and **Fuzzy Entropy**.

### 1. Shannon Entropy
**Description**: Measures the unpredictability of the signal’s amplitude distribution, like how varied a bag of candies is.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal to remove noise.
- **Step 2**: Create a histogram of signal amplitudes to estimate the probability distribution.
- **Step 3**: Compute Shannon Entropy using the formula: \( H = -\sum p_i \log_2(p_i) \), where \( p_i \) is the probability of each bin.
- **Tools**: `numpy` for histogram, `scipy.stats.entropy` for calculation.
- **Example**:
  - Input: ECG signal.
  - Process: Histogram, normalize, compute entropy.
  - Output: Shannon Entropy in bits.

**Code Example**:
```python
import numpy as np
from scipy.stats import entropy

# Simulated ECG signal (2 seconds, 360 Hz)
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Shannon Entropy
hist, bins = np.histogram(ecg_signal, bins=50, density=True)
hist = hist / np.sum(hist)  # Normalize to probability distribution
shannon_entropy = entropy(hist, base=2)
print("Shannon Entropy (bits):", shannon_entropy)
```

**Explanation**: The code creates a histogram of the ECG signal, normalizes it, and computes Shannon Entropy. For real data, load ECG signals using `wfdb`.

---

### 2. Approximate Entropy (ApEn)
**Description**: Measures signal regularity by checking how often patterns repeat, like noticing if a song’s rhythm is predictable.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Choose parameters: embedding dimension \( m \) (pattern length) and tolerance \( r \) (similarity threshold, typically 0.1–0.25 * std).
- **Step 3**: Compute ApEn by comparing the frequency of similar patterns of length \( m \) and \( m+1 \).
- **Tools**: `nolds.apen` for computation.
- **Example**:
  - Input: ECG signal.
  - Process: Compute ApEn with \( m=2 \), \( r=0.2*std \).
  - Output: ApEn value (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Approximate Entropy
apen = nolds.apen(ecg_signal, emb_dim=2, tolerance=0.2 * np.std(ecg_signal))
print("Approximate Entropy:", apen)
```

**Explanation**: The code uses `nolds.apen` to compute ApEn with an embedding dimension of 2 and a tolerance of 0.2 times the standard deviation.

---

### 3. Sample Entropy (SampEn)
**Description**: A more robust version of ApEn, less sensitive to signal length.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Choose parameters: embedding dimension \( m \), tolerance \( r \).
- **Step 3**: Compute SampEn by comparing pattern similarity, excluding self-matches.
- **Tools**: `nolds.sampen`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute SampEn with \( m=2 \), \( r=0.2*std \).
  - Output: SampEn value (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Sample Entropy
sampen = nolds.sampen(ecg_signal, emb_dim=2, tolerance=0.2 * np.std(ecg_signal))
print("Sample Entropy:", sampen)
```

**Explanation**: The code computes SampEn, which is similar to ApEn but more robust. Adjust `tolerance` for the signal.

---

### 4. Multiscale Entropy (MSE)
**Description**: Measures entropy at different time scales, like zooming in and out on a map.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply coarse-graining to create signals at different scales (e.g., average over 2, 3, 4 samples).
- **Step 3**: Compute SampEn for each scale.
- **Tools**: Custom implementation or `nolds` with a loop.
- **Example**:
  - Input: ECG signal.
  - Process: Coarse-grain signal, compute SampEn per scale.
  - Output: MSE values for multiple scales.

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Multiscale Entropy
mse = []
max_scale = 5
for scale in range(1, max_scale + 1):
    # Coarse-graining
    coarse_signal = np.mean(ecg_signal[:len(ecg_signal)//scale*scale].reshape(-1, scale), axis=1)
    mse.append(nolds.sampen(coarse_signal, emb_dim=2, tolerance=0.2 * np.std(coarse_signal)))
print("Multiscale Entropy (Scales 1-5):", mse)
```

**Explanation**: The code coarse-grains the signal by averaging over increasing window sizes and computes SampEn for each scale.

---

### 5. Permutation Entropy
**Description**: Measures complexity based on the order of signal values, like checking the sequence of dance moves.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Convert the signal into ordinal patterns (e.g., rank sequences of \( n \) samples).
- **Step 3**: Compute entropy of the pattern distribution.
- **Tools**: `nolds.perm_entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute permutation entropy with order \( n=3 \).
  - Output: Permutation Entropy (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Permutation Entropy
perm_entropy = nolds.perm_entropy(ecg_signal, order=3)
print("Permutation Entropy:", perm_entropy)
```

**Explanation**: The code computes permutation entropy with an order of 3, analyzing the sequence of signal values.

---

### 6. Spectral Entropy
**Description**: Measures the randomness of the signal’s frequency spectrum, like checking how varied a music playlist is.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Compute the power spectral density (PSD) using FFT.
- **Step 3**: Normalize the PSD and compute Shannon Entropy.
- **Tools**: `scipy.fft`, `scipy.stats.entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute PSD, normalize, compute entropy.
  - Output: Spectral Entropy (unitless).

**Code Example**:
```python
import numpy as np
from scipy.fft import fft
from scipy.stats import entropy

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Spectral Entropy
freqs = np.fft.fftfreq(len(ecg_signal), 1/sampling_rate)
psd = np.abs(fft(ecg_signal))**2
psd = psd / np.sum(psd)  # Normalize
spectral_entropy = entropy(psd, base=2)
print("Spectral Entropy (bits):", spectral_entropy)
```

**Explanation**: The code computes the FFT, derives the PSD, normalizes it, and calculates Shannon Entropy.

---

### 7. Wavelet Entropy
**Description**: Measures the randomness of wavelet coefficients across scales.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Compute Continuous Wavelet Transform (CWT).
- **Step 3**: Normalize squared coefficients and compute Shannon Entropy.
- **Tools**: `pywt.cwt`, `scipy.stats.entropy`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute CWT, normalize coefficients, compute entropy.
  - Output: Wavelet Entropy (unitless).

**Code Example**:
```python
import numpy as np
import pywt
from scipy.stats import entropy

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Wavelet Entropy
scales = np.arange(1, 64)
cwt_matrix, _ = pywt.cwt(ecg_signal, scales, 'morl', sampling_period=1/sampling_rate)
cwt_normalized = np.abs(cwt_matrix)**2 / np.sum(np.abs(cwt_matrix)**2)
wavelet_entropy = entropy(cwt_normalized.flatten(), base=2)
print("Wavelet Entropy (bits):", wavelet_entropy)
```

**Explanation**: The code computes CWT, normalizes squared coefficients, and calculates entropy.

---

### 8. Renyi Entropy
**Description**: A generalized entropy measure, adjustable with a parameter \( \alpha \).
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Create a histogram of signal amplitudes.
- **Step 3**: Compute Renyi Entropy: \( H_\alpha = \frac{1}{1-\alpha} \log_2 \left( \sum p_i^\alpha \right) \).
- **Tools**: `numpy` for histogram and calculation.
- **Example**:
  - Input: ECG signal.
  - Process: Histogram, compute Renyi Entropy with \( \alpha=2 \).
  - Output: Renyi Entropy (unitless).

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Renyi Entropy (alpha=2)
hist, bins = np.histogram(ecg_signal, bins=50, density=True)
hist = hist / np.sum(hist)
alpha = 2
renyi_entropy = (1 / (1 - alpha)) * np.log2(np.sum(hist**alpha))
print("Renyi Entropy (alpha=2, bits):", renyi_entropy)
```

**Explanation**: The code computes Renyi Entropy with \( \alpha=2 \). Adjust \( \alpha \) for different sensitivities.

---

### 9. Tsallis Entropy
**Description**: Another generalized entropy, useful for non-linear systems.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Create a histogram of signal amplitudes.
- **Step 3**: Compute Tsallis Entropy: \( H_q = \frac{1}{q-1} \left( 1 - \sum p_i^q \right) \).
- **Tools**: `numpy` for histogram and calculation.
- **Example**:
  - Input: ECG signal.
  - Process: Histogram, compute Tsallis Entropy with \( q=2 \).
  - Output: Tsallis Entropy (unitless).

**Code Example**:
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Tsallis Entropy (q=2)
hist, bins = np.histogram(ecg_signal, bins=50, density=True)
hist = hist / np.sum(hist)
q = 2
tsallis_entropy = (1 / (q - 1)) * (1 - np.sum(hist**q))
print("Tsallis Entropy (q=2):", tsallis_entropy)
```

**Explanation**: The code computes Tsallis Entropy with \( q=2 \). Adjust \( q \) for different analyses.

---

### 10. Fuzzy Entropy
**Description**: Measures signal irregularity using fuzzy logic, like judging a drawing’s complexity with flexible rules.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Choose parameters: embedding dimension \( m \), tolerance \( r \), fuzzy function (e.g., exponential).
- **Step 3**: Compute Fuzzy Entropy by comparing fuzzy similarities of patterns.
- **Tools**: Custom implementation or `EntropyHub` (not used here for simplicity).
- **Example**:
  - Input: ECG signal.
  - Process: Compute Fuzzy Entropy with \( m=2 \), \( r=0.2*std \).
  - Output: Fuzzy Entropy (unitless).

**Code Example** (Simplified, using a basic implementation):
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Simplified Fuzzy Entropy
def fuzzy_entropy(signal, m=2, r=0.2):
    n = len(signal)
    std = np.std(signal)
    r = r * std
    phi_m, phi_m1 = 0, 0
    for i in range(n - m):
        template = signal[i:i+m]
        count_m, count_m1 = 0, 0
        for j in range(n - m):
            if i != j:
                dist = np.max(np.abs(template - signal[j:j+m]))
                if dist <= r:
                    count_m += np.exp(-dist**2 / (r**2))
                if j < n - m - 1:
                    dist_m1 = np.max(np.abs(signal[i:i+m+1] - signal[j:j+m+1]))
                    if dist_m1 <= r:
                        count_m1 += np.exp(-dist_m1**2 / (r**2))
        phi_m += count_m / (n - m - 1)
        phi_m1 += count_m1 / (n - m - 2)
    return np.log(phi_m / phi_m1) if phi_m1 != 0 else np.inf

fuzzy_ent = fuzzy_entropy(ecg_signal, m=2, r=0.2)
print("Fuzzy Entropy:", fuzzy_ent)
```

**Explanation**: The code implements a basic Fuzzy Entropy calculation, comparing pattern similarities with a fuzzy membership function. For robust implementations, consider `EntropyHub`.

---

## 6.8 Nonlinear Features: Extraction Techniques

Nonlinear features capture complex, chaotic patterns in the ECG signal. Below are the extraction techniques for **Lyapunov Exponents**, **Correlation Dimension**, **Fractal Dimension**, **Hurst Exponent**, **Detrended Fluctuation Analysis (DFA)**, **Poincaré Plot Features**, **Recurrence Quantification Analysis (RQA)**, **Approximate Entropy**, **Sample Entropy**, and **Complexity Index**.

### 1. Lyapunov Exponents
**Description**: Measure how fast signal patterns diverge, indicating chaos.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Reconstruct the phase space with embedding dimension and lag.
- **Step 3**: Compute the largest Lyapunov exponent by tracking divergence rates.
- **Tools**: `nolds.lyap_r`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute largest Lyapunov exponent with \( emb_dim=10 \), \( lag=1 \).
  - Output: Lyapunov Exponent (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Lyapunov Exponent
lyap_exp = nolds.lyap_r(ecg_signal, emb_dim=10, lag=1, min_tsep=10)
print("Largest Lyapunov Exponent:", lyap_exp)
```

**Explanation**: The code computes the largest Lyapunov exponent, indicating chaotic behavior.

---

### 2. Correlation Dimension
**Description**: Estimates the complexity of the signal’s dynamics.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Reconstruct the phase space.
- **Step 3**: Compute the correlation dimension by analyzing point correlations.
- **Tools**: `nolds.corr_dim`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute correlation dimension with \( emb_dim=10 \).
  - Output: Correlation Dimension (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Correlation Dimension
corr_dim = nolds.corr_dim(ecg_signal, emb_dim=10)
print("Correlation Dimension:", corr_dim)
```

**Explanation**: The code computes the correlation dimension, reflecting dynamic complexity.

---

### 3. Fractal Dimension
**Description**: Measures the signal’s self-similarity or “roughness.”
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Apply a method like Higuchi’s algorithm to estimate fractal dimension.
- **Step 3**: Compute the dimension based on signal scaling properties.
- **Tools**: `nolds.hurst_rs` (as a proxy for fractal properties).
- **Example**:
  - Input: ECG signal.
  - Process: Compute fractal dimension via Higuchi’s method (simplified here).
  - Output: Fractal Dimension (unitless).

**Code Example** (Using Hurst as a proxy):
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Fractal Dimension (using Hurst Exponent as proxy)
hurst = nolds.hurst_rs(ecg_signal)
fractal_dim = 2 - hurst  # Approximate relation
print("Fractal Dimension (approx):", fractal_dim)
```

**Explanation**: The code uses the Hurst Exponent to estimate fractal dimension. For true fractal dimension, implement Higuchi’s algorithm.

---

### 4. Hurst Exponent
**Description**: Indicates long-term memory or trends in the signal.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Compute the Hurst Exponent using rescaled range analysis.
- **Tools**: `nolds.hurst_rs`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute Hurst Exponent.
  - Output: Hurst Exponent (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Hurst Exponent
hurst = nolds.hurst_rs(ecg_signal)
print("Hurst Exponent:", hurst)
```

**Explanation**: The code computes the Hurst Exponent, where values >0.5 indicate persistence.

---

### 5. Detrended Fluctuation Analysis (DFA)
**Description**: Measures self-similarity across scales.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Integrate the signal, divide into segments, and detrend.
- **Step 3**: Compute the fluctuation exponent.
- **Tools**: `nolds.dfa`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute DFA exponent.
  - Output: DFA exponent (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract DFA
dfa = nolds.dfa(ecg_signal)
print("DFA Exponent:", dfa)
```

**Explanation**: The code computes the DFA exponent, indicating self-similarity.

---

### 6. Poincaré Plot Features
**Description**: Quantifies variability in RR intervals using a scatter plot.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal and detect R peaks.
- **Step 2**: Compute RR intervals.
- **Step 3**: Calculate SD1 (short-term variability) and SD2 (long-term variability) from the Poincaré plot.
- **Tools**: `neurokit2.hrv_nonlinear`.
- **Example**:
  - Input: ECG signal.
  - Process: Compute RR intervals, extract SD1 and SD2.
  - Output: SD1, SD2 in milliseconds.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Poincaré Plot Features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
poincare = nk.hrv_nonlinear(info['ECG_R_Peaks'], sampling_rate=sampling_rate)
sd1 = poincare['HRV_SD1'].iloc[0]
sd2 = poincare['HRV_SD2'].iloc[0]
print("Poincaré SD1 (ms):", sd1)
print("Poincaré SD2 (ms):", sd2)
```

**Explanation**: The code computes SD1 and SD2 from RR intervals using `neurokit2`.

---

### 7. Recurrence Quantification Analysis (RQA)
**Description**: Analyzes repeating patterns in the signal’s phase space.
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Reconstruct the phase space.
- **Step 3**: Compute RQA metrics (e.g., recurrence rate, determinism).
- **Tools**: `pyRQA` or custom implementation (simplified here).
- **Example**:
  - Input: ECG signal.
  - Process: Compute recurrence rate.
  - Output: RQA metrics (unitless).

**Code Example** (Simplified):
```python
import numpy as np

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Simplified RQA (Recurrence Rate)
def simple_rqa(signal, m=3, tau=1, eps=0.1):
    n = len(signal)
    phase_space = np.array([signal[i:i+m:tau] for i in range(n-m+1)])
    dist = np.linalg.norm(phase_space[:, None] - phase_space, axis=2)
    recurrence_matrix = dist < eps
    recurrence_rate = np.sum(recurrence_matrix) / (n-m+1)**2
    return recurrence_rate

rqa_rr = simple_rqa(ecg_signal, m=3, tau=1, eps=0.1 * np.std(ecg_signal))
print("RQA Recurrence Rate:", rqa_rr)
```

**Explanation**: The code reconstructs the phase space and computes the recurrence rate. Use `pyRQA` for full RQA metrics.

---

### 8 & 9. Approximate Entropy and Sample Entropy
**Note**: Already covered under Entropy-Based Features (see above).

---

### 10. Complexity Index
**Description**: Combines multiple nonlinear measures (e.g., entropy, Lyapunov exponent).
**Extraction Technique**:
- **Step 1**: Clean the ECG signal.
- **Step 2**: Compute multiple nonlinear features (e.g., SampEn, Lyapunov exponent).
- **Step 3**: Combine (e.g., weighted sum or average) into a single index.
- **Tools**: `nolds`, custom combination.
- **Example**:
  - Input: ECG signal.
  - Process: Compute SampEn and Lyapunov exponent, average them.
  - Output: Complexity Index (unitless).

**Code Example**:
```python
import numpy as np
import nolds

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract Complexity Index
sampen = nolds.sampen(ecg_signal, emb_dim=2, tolerance=0.2 * np.std(ecg_signal))
lyap_exp = nolds.lyap_r(ecg_signal, emb_dim=10, lag=1, min_tsep=10)
complexity_index = (sampen + lyap_exp) / 2
print("Complexity Index:", complexity_index)
```

**Explanation**: The code averages SampEn and Lyapunov exponent for a simple complexity index.

---

## 6.9 Dimensionality Reduction Techniques: Extraction Techniques

Dimensionality reduction simplifies a large set of features. Below are the extraction techniques for **PCA**, **t-SNE**, **LDA**, **ICA**, **Autoencoders**, **UMAP**, **Factor Analysis**, **NMF**, **Isomap**, and **MDS**.

### 1. Principal Component Analysis (PCA)
**Description**: Projects features onto directions with maximum variance.
**Extraction Technique**:
- **Step 1**: Standardize features (zero mean, unit variance).
- **Step 2**: Compute PCA to get principal components.
- **Step 3**: Project features onto the top components.
- **Tools**: `sklearn.decomposition.PCA`.
- **Example**:
  - Input: Feature matrix (e.g., RR intervals, QRS amplitudes).
  - Process: Apply PCA to reduce to 2 components.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)
print("PCA Features Shape:", pca_features.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

**Explanation**: The code standardizes features and applies PCA to reduce to 2 components.

---

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
**Description**: Maps features to a low-dimensional space, preserving local relationships.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply t-SNE to reduce dimensions.
- **Tools**: `sklearn.manifold.TSNE`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply t-SNE to reduce to 2 dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract t-SNE
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features_scaled)
print("t-SNE Features Shape:", tsne_features.shape)
```

**Explanation**: The code applies t-SNE for visualization in 2D.

---

### 3. Linear Discriminant Analysis (LDA)
**Description**: Finds directions that best separate classes.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply LDA with class labels.
- **Step 3**: Project features onto discriminant axes.
- **Tools**: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
- **Example**:
  - Input: Feature matrix, class labels.
  - Process: Apply LDA to reduce dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)  # Simulated labels

# Extract LDA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
lda = LinearDiscriminantAnalysis(n_components=1)
lda_features = lda.fit_transform(features_scaled, target)
print("LDA Features Shape:", lda_features.shape)
```

**Explanation**: The code applies LDA with simulated labels to reduce features.

---

### 4. Independent Component Analysis (ICA)
**Description**: Separates mixed signals into independent sources.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply ICA to extract independent components.
- **Tools**: `sklearn.decomposition.FastICA`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply ICA to reduce dimensions.
  - Output: Independent components.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract ICA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
ica = FastICA(n_components=2, random_state=42)
ica_features = ica.fit_transform(features_scaled)
print("ICA Features Shape:", ica_features.shape)
```

**Explanation**: The code applies ICA to extract independent components.

---

### 5. Autoencoders
**Description**: Neural networks that compress data into a smaller representation.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Train an autoencoder to compress and reconstruct features.
- **Step 3**: Use the encoder part to extract reduced features.
- **Tools**: `tensorflow` or `keras`.
- **Example**:
  - Input: Feature matrix.
  - Process: Train autoencoder, extract encoded features.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract Autoencoder Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
input_dim = features_scaled.shape[1]
encoding_dim = 2
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='linear')(encoder)
autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=32, verbose=0)
encoder_model = Model(input_layer, encoder)
autoencoder_features = encoder_model.predict(features_scaled)
print("Autoencoder Features Shape:", autoencoder_features.shape)
```

**Explanation**: The code trains a simple autoencoder to reduce features to 2 dimensions.

---

### 6. Uniform Manifold Approximation and Projection (UMAP)
**Description**: Preserves data structure in a low-dimensional space, faster than t-SNE.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply UMAP to reduce dimensions.
- **Tools**: `umap-learn`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply UMAP to reduce to 2 dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import umap

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract UMAP
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_features = umap_model.fit_transform(features_scaled)
print("UMAP Features Shape:", umap_features.shape)
```

**Explanation**: The code applies UMAP to reduce features to 2 dimensions. Install `umap-learn` with `pip install umap-learn`.

---

### 7. Factor Analysis
**Description**: Identifies underlying factors explaining data variability.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply Factor Analysis to extract factors.
- **Tools**: `sklearn.decomposition.FactorAnalysis`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply Factor Analysis to reduce dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract Factor Analysis
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
fa = FactorAnalysis(n_components=2, random_state=42)
fa_features = fa.fit_transform(features_scaled)
print("Factor Analysis Features Shape:", fa_features.shape)
```

**Explanation**: The code applies Factor Analysis to extract factors.

---

### 8. Non-negative Matrix Factorization (NMF)
**Description**: Decomposes data into non-negative components.
**Extraction Technique**:
- **Step 1**: Ensure features are non-negative (e.g., scale if needed).
- **Step 2**: Apply NMF to reduce dimensions.
- **Tools**: `sklearn.decomposition.NMF`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply NMF to reduce to 2 components.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract NMF
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
nmf = NMF(n_components=2, random_state=42)
nmf_features = nmf.fit_transform(features_scaled)
print("NMF Features Shape:", nmf_features.shape)
```

**Explanation**: The code scales features to non-negative and applies NMF.

---

### 9. Isomap
**Description**: Preserves geodesic distances in the data manifold.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply Isomap to reduce dimensions.
- **Tools**: `sklearn.manifold.Isomap`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply Isomap to reduce to 2 dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract Isomap
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
isomap = Isomap(n_components=2)
isomap_features = isomap.fit_transform(features_scaled)
print("Isomap Features Shape:", isomap_features.shape)
```

**Explanation**: The code applies Isomap to reduce features to 2 dimensions.

---

### 10. Multidimensional Scaling (MDS)
**Description**: Maps data to a lower-dimensional space while preserving distances.
**Extraction Technique**:
- **Step 1**: Standardize features.
- **Step 2**: Apply MDS to reduce dimensions.
- **Tools**: `sklearn.manifold.MDS`.
- **Example**:
  - Input: Feature matrix.
  - Process: Apply MDS to reduce to 2 dimensions.
  - Output: Reduced feature matrix.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))

# Extract MDS
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
mds = MDS(n_components=2, random_state=42)
mds_features = mds.fit_transform(features_scaled)
print("MDS Features Shape:", mds_features.shape)
```

**Explanation**: The code applies MDS to reduce features to 2 dimensions.

---

## 6.10 Feature Selection Methods for ECG Analysis: Extraction Techniques

Feature selection picks the most relevant features for analysis. Below are the extraction techniques for **Filter Methods**, **Wrapper Methods**, **Embedded Methods**, **Mutual Information**, **Chi-Square Test**, **ANOVA F-Test**, **Recursive Feature Elimination (RFE)**, **Lasso (L1 Regularization)**, **Random Forest Feature Importance**, and **ReliefF Algorithm**.

### 1. Filter Methods
**Description**: Rank features using statistical measures (e.g., correlation).
**Extraction Technique**:
- **Step 1**: Compute a statistical metric (e.g., Pearson correlation) for each feature with the target.
- **Step 2**: Select top-ranked features.
- **Tools**: `numpy.corrcoef`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Compute correlations, select top features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Filter Method (Correlation)
correlations = [np.abs(np.corrcoef(features[:, i], target)[0, 1]) for i in range(features.shape[1])]
selected_features = np.argsort(correlations)[-2:]  # Top 2 features
print("Selected Feature Indices (Correlation):", selected_features)
```

**Explanation**: The code computes correlations and selects top features.

---

### 2. Wrapper Methods
**Description**: Test feature subsets with a model to find the best combination.
**Extraction Technique**:
- **Step 1**: Define a model (e.g., Random Forest).
- **Step 2**: Evaluate feature subsets iteratively (e.g., forward selection).
- **Tools**: Custom implementation (simplified here).
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Select features using forward selection.
  - Output: Selected feature indices.

**Code Example** (Simplified):
```python
import numpy as np
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Wrapper Method (Forward Selection)
model = RandomForestClassifier(random_state=42)
best_features = []
best_score = 0
remaining = list(range(features.shape[1]))
for _ in range(2):  # Select 2 features
    scores = []
    for i in remaining:
        current = best_features + [i]
        model.fit(features[:, current], target)
        score = accuracy_score(target, model.predict(features[:, current]))
        scores.append(score)
    best_idx = remaining[np.argmax(scores)]
    best_features.append(best_idx)
    remaining.remove(best_idx)
print("Selected Feature Indices (Wrapper):", best_features)
```

**Explanation**: The code performs forward selection to choose the best feature subset.

---

### 3. Embedded Methods
**Description**: Use models that inherently select features (e.g., Random Forest importance).
**Extraction Technique**:
- **Step 1**: Train a model that provides feature importance.
- **Step 2**: Select top features based on importance.
- **Tools**: `sklearn.ensemble.RandomForestClassifier`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Train Random Forest, select top features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Embedded Method
rf = RandomForestClassifier(random_state=42)
rf.fit(features, target)
importances = rf.feature_importances_
selected_features = np.argsort(importances)[-2:]  # Top 2 features
print("Selected Feature Indices (Embedded):", selected_features)
```

**Explanation**: The code uses Random Forest importances to select features.

---

### 4. Mutual Information
**Description**: Measures how much information a feature provides about the target.
**Extraction Technique**:
- **Step 1**: Compute mutual information between each feature and the target.
- **Step 2**: Select top-ranked features.
- **Tools**: `sklearn.feature_selection.mutual_info_classif`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Compute mutual information, select top features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.feature_selection import mutual_info_classif

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Mutual Information
mi_scores = mutual_info_classif(features, target)
selected_features = np.argsort(mi_scores)[-2:]  # Top 2 features
print("Selected Feature Indices (Mutual Information):", selected_features)
```

**Explanation**: The code computes mutual information and selects top features.

---

### 5. Chi-Square Test
**Description**: Tests feature independence for categorical targets.
**Extraction Technique**:
- **Step 1**: Ensure features are non-negative (e.g., scale).
- **Step 2**: Compute Chi-Square statistic.
- **Step 3**: Select top-ranked features.
- **Tools**: `sklearn.feature_selection.chi2`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Compute Chi-Square, select top features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Chi-Square
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
chi2_scores, _ = chi2(features_scaled, target)
selected_features = np.argsort(chi2_scores)[-2:]  # Top 2 features
print("Selected Feature Indices (Chi-Square):", selected_features)
```

**Explanation**: The code computes Chi-Square statistics and selects top features.

---

### 6. ANOVA F-Test
**Description**: Tests feature significance for continuous features and categorical targets.
**Extraction Technique**:
- **Step 1**: Compute F-statistic for each feature.
- **Step 2**: Select top-ranked features.
- **Tools**: `sklearn.feature_selection.f_classif`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Compute F-test, select top features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.feature_selection import f_classif

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract ANOVA F-Test
f_scores, _ = f_classif(features, target)
selected_features = np.argsort(f_scores)[-2:]  # Top 2 features
print("Selected Feature Indices (ANOVA F-Test):", selected_features)
```

**Explanation**: The code computes F-statistics and selects top features.

---

### 7. Recursive Feature Elimination (RFE)
**Description**: Iteratively removes least important features using a model.
**Extraction Technique**:
- **Step 1**: Train a model (e.g., Random Forest).
- **Step 2**: Use RFE to recursively eliminate features.
- **Tools**: `sklearn.feature_selection.RFE`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Apply RFE with Random Forest.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract RFE
rf = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=2)
rfe.fit(features, target)
selected_features = np.where(rfe.support_)[0]
print("Selected Feature Indices (RFE):", selected_features)
```

**Explanation**: The code uses RFE with a Random Forest model to select features.

---

### 8. Lasso (L1 Regularization)
**Description**: Shrinks unimportant feature coefficients to zero.
**Extraction Technique**:
- **Step 1**: Train a Lasso model.
- **Step 2**: Select features with non-zero coefficients.
- **Tools**: `sklearn.linear_model.Lasso`.
- **Example**:
  - Input: Feature matrix, target labels.
  - Process: Apply Lasso, select non-zero features.
  - Output: Selected feature indices.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Simulated ECG signal
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len]))
target = np.random.randint(0, 2, min_len)

# Extract Lasso
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(features_scaled, target)
selected_features = np.where(lasso.coef_ != 0)[0]
print("Selected Feature Indices (Lasso):", selected_features)
```

**Explanation**: The code applies Lasso regression to select features with non-zero coefficients.

---

### 9. Random Forest Feature Importance

**Description**: Random Forest Feature Importance measures how much each feature contributes to the accuracy of a Random Forest model, like figuring out which ingredients make a cake taste the best. Features that improve the model's ability to classify ECG signals (e.g., normal vs. abnormal) get higher importance scores.

**Extraction Technique**:
- **Step 1**: Prepare a feature matrix (e.g., RR intervals, QRS amplitudes) and target labels (e.g., normal or abnormal ECG).
- **Step 2**: Train a Random Forest Classifier on the feature matrix and labels.
- **Step 3**: Extract feature importance scores from the trained model.
- **Step 4**: Select the top-ranked features based on their importance scores.
- **Tools**: `sklearn.ensemble.RandomForestClassifier` for training and extracting importances.
- **Example**:
  - **Input**: Feature matrix (e.g., RR intervals, QRS amplitudes), binary target labels.
  - **Process**: Train Random Forest, extract importance scores, select top 2 features.
  - **Output**: Indices of the most important features.

**Code Example** (Completing the previous artifact):
```python
import numpy as np
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier

# Simulated ECG signal (2 seconds, 360 Hz)
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
variance = np.var(ecg_signal)
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len], np.repeat(variance, min_len)))
target = np.random.randint(0, 2, min_len)  # Simulated labels (0=normal, 1=abnormal)

# Extract Random Forest Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(features, target)
importances = rf.feature_importances_
selected_features = np.argsort(importances)[-2:]  # Select top 2 features
print("Random Forest Importances:", importances)
print("Selected Feature Indices (Random Forest):", selected_features)
```

**Explanation**:
- **Setup**: The code generates a simulated ECG signal (sine wave with noise) at 360 Hz, mimicking a real ECG. It extracts features like RR intervals (time between heartbeats), QRS amplitudes, and signal variance.
- **Feature Matrix**: Combines features into a matrix where each row is a sample and columns are features (RR intervals, QRS amplitudes, variance).
- **Random Forest**: Trains a Random Forest Classifier, which assigns importance scores to each feature based on how much they improve classification accuracy.
- **Output**: Prints the importance scores and selects the indices of the top 2 features (e.g., [0, 1] for RR intervals and QRS amplitudes if they’re most important).
- **Practical Note**: For real ECG data, use Physionet’s MIT-BIH Arrhythmia Database with `wfdb` to load labeled signals. Standardize features (e.g., using `StandardScaler`) for better performance.

**Analogy**: Imagine you’re baking a cake and trying different ingredients (features). The Random Forest tells you which ingredients (e.g., sugar, flour) make the cake taste best by scoring their impact on the final flavor.

---

### 10. ReliefF Algorithm

**Description**: The ReliefF Algorithm ranks features based on their ability to distinguish between classes (e.g., normal vs. abnormal ECGs) by comparing nearby samples, like picking the best players for a soccer team by seeing who performs best in practice games.

**Extraction Technique**:
- **Step 1**: Prepare a feature matrix and target labels.
- **Step 2**: Apply the ReliefF algorithm, which assigns weights to features based on how well they differentiate between classes in nearby samples.
- **Step 3**: Select the top-ranked features based on their weights.
- **Tools**: `skrebate.ReliefF` for computation (requires installation via `pip install skrebate`).
- **Example**:
  - **Input**: Feature matrix (e.g., RR intervals, QRS amplitudes, variance), binary target labels.
  - **Process**: Compute ReliefF weights, select top 2 features.
  - **Output**: Indices of the most important features.

**Code Example**:
```python
import numpy as np
import neurokit2 as nk
from skrebate import ReliefF

# Simulated ECG signal (2 seconds, 360 Hz)
sampling_rate = 360
time = np.arange(0, 2, 1/sampling_rate)
ecg_signal = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(len(time))

# Extract features
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
r_peaks = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks) / sampling_rate
qrs_amplitudes = ecg_cleaned[r_peaks]
variance = np.var(ecg_signal)
min_len = min(len(rr_intervals), len(qrs_amplitudes))
features = np.column_stack((rr_intervals[:min_len], qrs_amplitudes[:min_len], np.repeat(variance, min_len)))
target = np.random.randint(0, 2, min_len)  # Simulated labels (0=normal, 1=abnormal)

# Extract ReliefF
relieff = ReliefF(n_features_to_select=2, n_neighbors=10)
relieff.fit(features, target)
selected_features = np.argsort(relieff.feature_importances_)[-2:]  # Top 2 features
print("ReliefF Feature Importances:", relieff.feature_importances_)
print("Selected Feature Indices (ReliefF):", selected_features)
```

**Explanation**:
- **Setup**: Generates a simulated ECG signal and extracts features (RR intervals, QRS amplitudes, variance).
- **Feature Matrix**: Combines features into a matrix.
- **ReliefF**: Applies the ReliefF algorithm, which weights features based on their ability to distinguish between classes by comparing nearby samples. The `n_neighbors=10` parameter sets the number of neighbors to consider.
- **Output**: Prints the ReliefF importance scores and selects the top 2 features.
- **Practical Note**: Install `skrebate` with `pip install skrebate`. For real data, use labeled ECG datasets and ensure features are standardized. ReliefF is robust to noise and works well with small datasets.

**Analogy**: Think of ReliefF as a coach picking players by watching how well they perform compared to their teammates in practice. Features that consistently help distinguish normal from abnormal ECGs get higher scores.

---

## Summary of Extraction Techniques

Here’s a quick recap of all the extraction techniques covered for the listed features, ensuring every aspect is addressed:

### 6.7 Entropy-Based Features
1. **Shannon Entropy**: Histogram of signal amplitudes, compute \( H = -\sum p_i \log_2(p_i) \).
2. **Approximate Entropy (ApEn)**: Compare pattern similarities with embedding dimension \( m \) and tolerance \( r \).
3. **Sample Entropy (SampEn)**: Similar to ApEn but excludes self-matches, more robust.
4. **Multiscale Entropy (MSE)**: Coarse-grain signal at multiple scales, compute SampEn per scale.
5. **Permutation Entropy**: Analyze ordinal patterns of signal values.
6. **Spectral Entropy**: Compute FFT, normalize PSD, apply Shannon Entropy.
7. **Wavelet Entropy**: Compute CWT, normalize coefficients, apply Shannon Entropy.
8. **Renyi Entropy**: Histogram, compute \( H_\alpha = \frac{1}{1-\alpha} \log_2 \left( \sum p_i^\alpha \right) \).
9. **Tsallis Entropy**: Histogram, compute \( H_q = \frac{1}{q-1} \left( 1 - \sum p_i^q \right) \).
10. **Fuzzy Entropy**: Compare pattern similarities using fuzzy membership functions.

### 6.8 Nonlinear Features
1. **Lyapunov Exponents**: Reconstruct phase space, compute divergence rate.
2. **Correlation Dimension**: Reconstruct phase space, analyze point correlations.
3. **Fractal Dimension**: Estimate using Higuchi’s algorithm or Hurst Exponent (as proxy).
4. **Hurst Exponent**: Compute rescaled range analysis.
5. **Detrended Fluctuation Analysis (DFA)**: Integrate signal, detrend, compute fluctuation exponent.
6. **Poincaré Plot Features**: Compute RR intervals, calculate SD1 and SD2.
7. **Recurrence Quantification Analysis (RQA)**: Reconstruct phase space, compute recurrence metrics.
8. **Approximate Entropy**: Covered under Entropy-Based Features.
9. **Sample Entropy**: Covered under Entropy-Based Features.
10. **Complexity Index**: Combine nonlinear features (e.g., SampEn, Lyapunov exponent).

### 6.9 Dimensionality Reduction Techniques
1. **Principal Component Analysis (PCA)**: Standardize features, project onto principal components.
2. **t-SNE**: Standardize features, map to low-dimensional space preserving local structure.
3. **Linear Discriminant Analysis (LDA)**: Standardize features, project onto discriminant axes.
4. **Independent Component Analysis (ICA)**: Standardize features, extract independent components.
5. **Autoencoders**: Train neural network to compress features.
6. **UMAP**: Standardize features, map to low-dimensional space preserving structure.
7. **Factor Analysis**: Standardize features, extract underlying factors.
8. **Non-negative Matrix Factorization (NMF)**: Scale features to non-negative, decompose into components.
9. **Isomap**: Standardize features, preserve geodesic distances.
10. **Multidimensional Scaling (MDS)**: Standardize features, preserve Euclidean distances.

### 6.10 Feature Selection Methods
1. **Filter Methods**: Rank features using statistical metrics (e.g., correlation).
2. **Wrapper Methods**: Evaluate feature subsets with a model (e.g., forward selection).
3. **Embedded Methods**: Use model-inherent feature selection (e.g., Random Forest importance).
4. **Mutual Information**: Compute information gain between features and target.
5. **Chi-Square Test**: Compute Chi-Square statistic for non-negative features.
6. **ANOVA F-Test**: Compute F-statistic for feature significance.
7. **Recursive Feature Elimination (RFE)**: Iteratively remove least important features using a model.
8. **Lasso (L1 Regularization)**: Train Lasso model, select features with non-zero coefficients.
9. **Random Forest Feature Importance**: Train Random Forest, select top features by importance.
10. **ReliefF Algorithm**: Weight features based on ability to distinguish classes in nearby samples.

---

## Practical Tips for Implementation

1. **Install Required Libraries**:
   ```bash
   pip install numpy scipy neurokit2 nolds scikit-learn tensorflow skrebate umap-learn pywt
   ```
   These cover all tools used in the examples.

2. **Use Real ECG Data**:
   - Download datasets from Physionet (e.g., MIT-BIH Arrhythmia Database) using `wfdb`:
     ```python
     import wfdb
     record = wfdb.rdrecord('mitdb/100', sampto=7200)
     ecg_signal = record.p_signal[:, 0]
     ```
   - Replace simulated signals in the code with real data.

3. **Preprocess Signals**:
   - Clean ECG signals using `neurokit2.ecg_clean` to remove noise (e.g., baseline wander, powerline interference).
   - Use a bandpass filter (0.5–40 Hz) for better feature extraction.

4. **Standardize Features**:
   - Always standardize (zero mean, unit variance) or scale (non-negative for some methods) features before dimensionality reduction or feature selection to ensure fair comparisons.

5. **Parameter Tuning**:
   - For entropy features (e.g., ApEn, SampEn), experiment with `emb_dim` (2–3) and `tolerance` (0.1–0.25 * std).
   - For nonlinear features like Lyapunov exponents, use longer signals (5–10 minutes) for accuracy.
   - For dimensionality reduction, try different `n_components` (e.g., 2–5).
   - For feature selection, adjust the number of selected features based on the model’s needs.

6. **Combine Features**:
   - Use a mix of entropy, nonlinear, and other features (e.g., morphological, time-frequency) for robust ECG analysis.
   - Feed selected or reduced features into machine learning models (e.g., SVM, Random Forest) for classification tasks like arrhythmia detection.

7. **Visualize Results**:
   - Plot ECG signals, Poincaré plots, or feature importance to verify extraction.
   - For dimensionality reduction, use scatter plots to visualize reduced features.

---

## How to Use These in the PhD Research

For the PhD in Biomedical Signal Processing:
- **Experiment with Features**: Test which entropy or nonlinear features best detect specific conditions (e.g., atrial fibrillation, ventricular tachycardia).
- **Optimize Dimensionality Reduction**: Use PCA or t-SNE to visualize high-dimensional ECG data, helping identify patterns in heart conditions.
- **Select Features Wisely**: Apply feature selection to reduce computational load and improve model accuracy for real-time ECG analysis.
- **Validate with Real Data**: Use labeled datasets (e.g., MIT-BIH, PTB Diagnostic ECG Database) to validate feature effectiveness.
- **Document Results**: Compare feature performance (e.g., classification accuracy) in the thesis to show their impact.
