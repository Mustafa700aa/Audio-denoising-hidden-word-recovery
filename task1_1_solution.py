"""
=============================================================================
TASK 1.1: UNCOVER THE SECRET HIDDEN WORD (AUDIO DE-NOISING SOLUTION)
=============================================================================
Course / Project: Signal Processing & AI Team Phase II
Author: AI Assistant Pair Programmer
Task: Recover a speech signal contaminated with loud buzz interference.

METHODOLOGY OVERVIEW:
1. Audio Ingestion & Metadata Inspection:
   - Reads the degraded WAV file (`task5_1.wav`).
   - Analyzes sampling rate (48 kHz), bit depth (16-bit PCM), and duration (~1.12 s).

2. Frequency Domain Spectral Analysis (FFT):
   - Computes the Discrete Fourier Transform via Fast Fourier Transform (FFT).
   - Identifies the sharp periodic noise spikes (fundamental at ~320 Hz and 
     modulation/harmonic tones at ~640 Hz / 2309 Hz intervals).

3. Digital Filter Design & Processing:
   - Cascade of 2nd-order IIR Notch Filters (`scipy.signal.iirnotch`):
     Surgically attenuates narrow buzz tones with high Quality factor (Q >= 30),
     preserving broadband speech formants.
   - Butterworth Bandpass Filter (80 Hz – 4000 Hz, 4th order):
     Removes out-of-band low rumble and high-frequency noise outside normal
     human vocal range.
   - Zero-Phase Filtering (`scipy.signal.filtfilt`):
     Applies forward-backward filtering to ensure linear phase (zero phase distortion),
     preventing speech smear and waveform temporal shifting.

4. Audio Normalization & Output:
   - Rescales the filtered speech to 16-bit integer range (-32767 to 32767).
   - Exports the cleaned audio to `cleaned_audio.wav`.

5. Verification & Visualization:
   - Produces a 3-panel diagnostic plot comparing Time-Domain Waveforms,
     FFT Magnitude Spectra (dB), and Spectrograms before and after filtering.

6. Audio Playback:
   - Plays the cleaned audio at the end of the script using `winsound` / `sounddevice`.
=============================================================================
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def load_audio(filepath: str):
    """
    Load a WAV audio file and return sample rate, raw data, and normalized float data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file '{filepath}' not found.")
    
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        # Convert stereo to mono by taking the first channel
        data = data[:, 0]
    
    data_float = data.astype(np.float64)
    duration = len(data) / sr
    
    print(f"[+] Loaded: {filepath}")
    print(f"    - Sample Rate : {sr} Hz")
    print(f"    - Channels    : 1 (Mono)")
    print(f"    - Total Samples: {len(data)}")
    print(f"    - Duration    : {duration:.3f} seconds")
    print(f"    - Dynamic Range: [{np.min(data)}, {np.max(data)}]")
    
    return sr, data, data_float


def detect_noise_frequencies(audio_float: np.ndarray, sr: int, prominence_ratio: float = 4.0):
    """
    Identify prominent sharp spectral spikes characteristic of buzz noise using FFT.
    """
    N = len(audio_float)
    freqs = np.fft.rfftfreq(N, 1 / sr)
    fft_mag = np.abs(np.fft.rfft(audio_float))
    
    # Estimate baseline spectrum using a median filter (~500 Hz window)
    window_bins = int(500 * N / sr) | 1
    baseline = median_filter(fft_mag, size=window_bins)
    
    # Calculate ratio of peak magnitude to local baseline
    spectral_ratio = fft_mag / (baseline + 1e-6)
    
    # Find peaks that exceed the baseline by the given prominence ratio
    peak_indices, _ = signal.find_peaks(
        spectral_ratio,
        height=prominence_ratio,
        distance=int(5 * N / sr)
    )
    
    detected_freqs = freqs[peak_indices]
    print(f"[+] Detected {len(detected_freqs)} distinct buzz interference frequencies.")
    return freqs, fft_mag, peak_indices, detected_freqs


def design_and_apply_filters(audio_float: np.ndarray, sr: int, detected_freqs: np.ndarray):
    """
    Apply a cascade of IIR notch filters and a bandpass filter using zero-phase filtering.
    """
    filtered_audio = audio_float.copy()
    
    # 1. Apply Cascade of IIR Notch Filters
    notch_count = 0
    for f0 in detected_freqs:
        # Avoid DC and Nyquist boundaries
        if 20 < f0 < (sr / 2 - 20):
            # Quality factor Q: higher Q gives a narrower notch to preserve speech
            Q = max(30.0, f0 / 5.0)
            b, a = signal.iirnotch(w0=f0, Q=Q, fs=sr)
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            notch_count += 1
            
    print(f"[+] Applied {notch_count} IIR Notch Filters (zero-phase filtfilt).")
    
    # 2. Apply Butterworth Bandpass Filter (80 Hz to 4000 Hz for speech)
    lowcut = 80.0
    highcut = min(4000.0, sr / 2 - 100.0)
    b_bp, a_bp = signal.butter(N=4, Wn=[lowcut, highcut], btype='bandpass', fs=sr)
    filtered_audio = signal.filtfilt(b_bp, a_bp, filtered_audio)
    print(f"[+] Applied 4th-Order Butterworth Bandpass Filter [{lowcut:.0f} Hz - {highcut:.0f} Hz].")
    
    return filtered_audio


def normalize_and_save(audio_float: np.ndarray, sr: int, output_path: str):
    """
    Normalize audio amplitude and save as 16-bit PCM WAV.
    """
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        normalized = (audio_float / max_val * 32767.0 * 0.95).astype(np.int16)
    else:
        normalized = audio_float.astype(np.int16)
    
    wavfile.write(output_path, sr, normalized)
    print(f"[+] Cleaned audio successfully saved to: {output_path}")
    return normalized


def plot_diagnostics(orig_data: np.ndarray, clean_data: np.ndarray, sr: int, save_path: str = "audio_filtering_analysis.png"):
    """
    Create a comprehensive 3-panel visualization comparing:
    1. Time-Domain Waveforms
    2. Frequency Spectra (FFT in dB)
    3. Spectrograms
    """
    t_orig = np.arange(len(orig_data)) / sr
    t_clean = np.arange(len(clean_data)) / sr
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    
    # --- 1. Time-Domain Waveforms ---
    axes[0, 0].plot(t_orig, orig_data, color='#c0392b', lw=0.8)
    axes[0, 0].set_title("Original Audio (Noisy & Clipped Buzz)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_ylim(-35000, 35000)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_clean, clean_data, color='#27ae60', lw=0.8)
    axes[0, 1].set_title("Cleaned Audio (Filtered & Recovered Speech)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- 2. Frequency Spectra (FFT) ---
    N_orig = len(orig_data)
    freqs_orig = np.fft.rfftfreq(N_orig, 1 / sr)
    fft_orig = np.abs(np.fft.rfft(orig_data.astype(float)))
    
    N_clean = len(clean_data)
    freqs_clean = np.fft.rfftfreq(N_clean, 1 / sr)
    fft_clean = np.abs(np.fft.rfft(clean_data.astype(float)))
    
    axes[1, 0].plot(freqs_orig, 20 * np.log10(fft_orig + 1e-6), color='#c0392b', lw=0.8)
    axes[1, 0].set_title("Original Frequency Spectrum (0 - 6000 Hz)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].set_xlim(0, 6000)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(freqs_clean, 20 * np.log10(fft_clean + 1e-6), color='#27ae60', lw=0.8)
    axes[1, 1].set_title("Cleaned Frequency Spectrum (0 - 6000 Hz)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].set_xlim(0, 6000)
    axes[1, 1].grid(True, alpha=0.3)
    
    # --- 3. Spectrograms ---
    f_o, t_o, S_o = signal.spectrogram(orig_data, sr, nperseg=512, noverlap=384)
    im0 = axes[2, 0].pcolormesh(t_o, f_o, 10 * np.log10(S_o + 1e-10), shading='gouraud', cmap='magma')
    axes[2, 0].set_title("Original Spectrogram (Periodic Buzz Masking Content)", fontsize=11, fontweight='bold')
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Frequency (Hz)")
    axes[2, 0].set_ylim(0, 6000)
    fig.colorbar(im0, ax=axes[2, 0], label='Power (dB)')
    
    f_c, t_c, S_c = signal.spectrogram(clean_data, sr, nperseg=512, noverlap=384)
    im1 = axes[2, 1].pcolormesh(t_c, f_c, 10 * np.log10(S_c + 1e-10), shading='gouraud', cmap='viridis')
    axes[2, 1].set_title("Cleaned Spectrogram (Speech Formants Cleared)", fontsize=11, fontweight='bold')
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    axes[2, 1].set_ylim(0, 6000)
    fig.colorbar(im1, ax=axes[2, 1], label='Power (dB)')
    
    plt.suptitle("DSP Audio De-Noising Analysis (Task 1.1)", fontsize=14, fontweight='bold', y=0.99)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Diagnostic plots saved to: {save_path}")


def play_audio(audio_path: str):
    """
    Play the cleaned audio file through system audio output.
    """
    print(f"\n[>] Playing cleaned audio: {audio_path} ...")
    
    # 1. Try Windows native winsound (no external dependencies)
    try:
        import winsound
        winsound.PlaySound(audio_path, winsound.SND_FILENAME)
        print("[+] Playback completed via Windows Sound API (winsound).")
        return
    except Exception as e:
        pass
    
    # 2. Try sounddevice
    try:
        import sounddevice as sd
        sr, data = wavfile.read(audio_path)
        sd.play(data, sr)
        sd.wait()
        print("[+] Playback completed via sounddevice.")
        return
    except Exception as e:
        pass
    
    # 3. Fallback: Open with default OS media player
    try:
        os.startfile(audio_path)
        print("[+] Launched audio in system default media player.")
    except Exception as e:
        print(f"[-] Could not trigger audio playback automatically: {e}")


def main():
    input_wav = "task5_1.wav"
    output_wav = "cleaned_audio.wav"
    plot_png = "audio_filtering_analysis.png"
    
    print("=" * 70)
    print("  TASK 1.1: AUDIO DE-NOISING & HIDDEN WORD RECOVERY")
    print("=" * 70)
    
    # Step 1: Load Audio
    sr, orig_data, audio_float = load_audio(input_wav)
    
    # Step 2: Spectral Analysis & Noise Peak Detection
    freqs, fft_mag, peak_indices, detected_freqs = detect_noise_frequencies(
        audio_float, sr, prominence_ratio=4.0
    )
    
    # Step 3: Design and Apply Digital Filters
    cleaned_float = design_and_apply_filters(audio_float, sr, detected_freqs)
    
    # Step 4: Normalize and Save Cleaned Audio
    cleaned_data = normalize_and_save(cleaned_float, sr, output_wav)
    
    # Step 5: Generate Verification & Diagnostic Visualizations
    plot_diagnostics(orig_data, cleaned_data, sr, plot_png)
    
    # Step 6: Audio Playback
    play_audio(output_wav)
    
    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"  - Cleaned audio file : {os.path.abspath(output_wav)}")
    print(f"  - Diagnostic plots   : {os.path.abspath(plot_png)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
