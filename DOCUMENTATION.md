# Task 1.1: Uncover the Secret Hidden Word — Technical Documentation

---

## 1. Executive Summary & Recovered Word

* **Input File**: [`task5_1.wav`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/task5_1.wav) (48 kHz, 16-bit Mono, ~1.117 seconds duration)
* **Recovered Audio**: [`cleaned_audio.wav`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/cleaned_audio.wav)
* **Solution Scripts**:
  * Python script: [`task1_1_solution.py`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/task1_1_solution.py)
  * Jupyter Notebook: [`Task_1_1_Solution.ipynb`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/Task_1_1_Solution.ipynb)
* **Diagnostic Visualization**: [`audio_filtering_analysis.png`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/audio_filtering_analysis.png)
* **Recovered Secret Word / Phrase**: **"Thank you"**

---

## 2. Problem Characterization

The original audio file contains speech masked by a severe, high-amplitude buzz noise.
- **Time-Domain Analysis**: The raw signal values swing aggressively between $+32767$ and $-32767$ (saturating $\sim 89\%$ of the samples), resembling a high-energy square wave / pulse interference.
- **Frequency-Domain Analysis (FFT)**: Fast Fourier Transform revealed that the buzz is composed of a fundamental tone at $\approx 319.55\text{ Hz}$ accompanied by an extensive periodic comb of harmonic and modulation spikes separated by $640\text{ Hz}$ and $2309\text{ Hz}$ intervals stretching across the entire spectrum up to $24\text{ kHz}$.

---

## 3. Engineering Approach (Step-by-Step)

Our DSP filtering pipeline follows four primary stages:

```
[task5_1.wav]
     │
     ▼
[FFT Spectral Analysis] ──► Detect buzz spike frequencies using local baseline ratio
     │
     ▼
[IIR Notch Filter Cascade] ──► Surgically attenuate each sharp buzz tone (Q >= 30)
     │
     ▼
[Butterworth Bandpass Filter] ──► Keep vocal frequency range (80 Hz - 4000 Hz)
     │
     ▼
[Zero-Phase Filtering (filtfilt)] ──► Prevent time delay & phase smear
     │
     ▼
[Normalization & Audio Playback] ──► Export cleaned_audio.wav & play via Windows Sound
```

### Step 1: Ingestion & Spectral Decomposition
We compute the Discrete Fourier Transform using `numpy.fft.rfft`. To separate the stationary buzz interference from the underlying speech formant spectrum, we compute a running baseline magnitude using median filtering. Peaks exceeding the local median by a prominence threshold are tagged as interference frequencies.

### Step 2: High-Q IIR Notch Filter Cascade
For each detected interference frequency $f_0$, we synthesize a 2nd-order IIR Notch Filter (`scipy.signal.iirnotch`).
- **Quality Factor ($Q \ge 30$)**: Controls the sharpness of the notch. A high $Q$ ensures the notch is extremely narrow (a few Hertz wide), rejecting the noise tone while leaving adjacent speech formants intact.

### Step 3: Butterworth Bandpass Preconditioning
Human speech energy and intelligibility reside predominantly between $80\text{ Hz}$ (fundamental vocal cord pitch) and $4000\text{ Hz}$ (upper fricative/consonant formants). A 4th-order Butterworth bandpass filter (`scipy.signal.butter`) removes residual low-frequency sub-audible hums and high-frequency noise spikes above $4\text{ kHz}$.

### Step 4: Zero-Phase Filtering (`filtfilt`)
Traditional IIR filters introduce non-linear phase shifts that alter the temporal alignment of frequency components, resulting in muffled speech. By using `scipy.signal.filtfilt`, the filter passes the signal forward and then backward, achieving **exact zero phase distortion** ($0^\circ$).

### Step 5: Normalization and Playback
The filtered floating-point signal is peak-normalized to $95\%$ of 16-bit dynamic range ($-32767$ to $+32767$), saved to `cleaned_audio.wav`, and played automatically using `winsound` / `sounddevice`.

---

## 4. Explanation of Tools Used (In Simple Terms)

| Tool / Function | Library | Purpose in Simple Words |
|---|---|---|
| `wavfile.read` / `wavfile.write` | `scipy.io` | Reads and writes the WAV audio files to and from raw numeric sample arrays. |
| `np.fft.rfft` / `np.fft.rfftfreq` | `numpy` | Converts the time-domain audio waveform into its frequency spectrum (like an equalizer display) so we can see the exact frequencies of the noise. |
| `signal.find_peaks` / `median_filter` | `scipy.signal` & `scipy.ndimage` | Automatically pinpoints the tall, sharp noise spikes above the normal voice background. |
| `signal.iirnotch` | `scipy.signal` | Creates a precision "notch" filter that acts like a laser, cutting out only one specific buzz tone without damaging the voice around it. |
| `signal.butter` | `scipy.signal` | Creates a smooth bandpass filter that allows standard speech frequencies ($80\text{ Hz} - 4000\text{ Hz}$) to pass while blocking everything outside. |
| `signal.filtfilt` | `scipy.signal` | Applies the filters both forwards and backwards so that the audio has zero time delay and no phase smearing. |
| `matplotlib.pyplot` | `matplotlib` | Plots the time-domain waveforms, frequency spectrum in decibels, and spectrograms to visually verify the cleanup. |
| `winsound` / `sounddevice` | Standard Library / `sounddevice` | Plays the cleaned audio directly through your speakers at the end of the script. |

---

## 5. Visual Comparison

The script generates [`audio_filtering_analysis.png`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/audio_filtering_analysis.png) which illustrates:
1. **Time Domain**: The original heavily clipped square-like waveform is restored to a natural, amplitude-modulated speech envelope.
2. **FFT Spectrum**: The tall noise spikes exceeding $160\text{ dB}$ are suppressed by over $40\text{ dB}$, revealing the smooth vocal spectral curve.
3. **Spectrogram**: The dense horizontal noise bars masking the spectrogram are eliminated, clearly uncovering the speech formants (vowel resonances and consonant transitions).

---

## 6. How to Run

### Option A: Python Script
```bash
python task1_1_solution.py
```
*Output*: De-noises audio, saves `cleaned_audio.wav`, produces `audio_filtering_analysis.png`, and automatically plays the audio through your speakers.

### Option B: Jupyter Notebook
Open and run all cells in [`Task_1_1_Solution.ipynb`](file:///c:/Users/ZBook%20G3/OneDrive/Desktop/MIA_Phase5/Task_1_1_Solution.ipynb) to view interactive inline audio players and plots.
