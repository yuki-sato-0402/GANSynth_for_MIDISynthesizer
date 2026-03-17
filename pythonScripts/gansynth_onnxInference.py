import onnxruntime as ort
import numpy as np
import librosa
import soundfile as sf
import os

def hertz_to_mel(frequencies_hertz):
    return 1127.0 * np.log(1.0 + (frequencies_hertz / 700.0))

def mel_to_hertz(mel_values):
    return 700.0 * (np.exp(mel_values / 1127.0) - 1.0)

def linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate):
    """Faithfully reproduce the logic of spectral_ops.py"""
    nyquist_hertz = sample_rate / 2.0
    #Exclude DC bin (GANSynth/HTK default)
    bands_to_zero = 1
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    
    band_edges_mel = np.linspace(hertz_to_mel(0.0), hertz_to_mel(nyquist_hertz), num_mel_bins + 2)
    
    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]
    
    # Ensuring minimum bandwidth
    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    print(f"Minimum frequency threshold for mel bands: {freq_th} Hz")
    for i in range(num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + 700.0)
            dm = 1127.0 * np.log(rhs + np.sqrt(1.0 + rhs**2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    lower_slopes = (linear_frequencies - lower_edge_hz) / (center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (upper_edge_hz - center_hz)
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    
    # Return the DC bin portion with padding [1025, 1024]
    mel_weights_matrix = np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], 'constant')
    return mel_weights_matrix

def get_mel_to_linear_matrix(n_mel, n_mag, sr):
    """Replicate _mel_to_linear_matrix from specgrams_helper.py"""
    m = linear_to_mel_weight_matrix(n_mel, n_mag, sr)
    m_t = m.T
    p = np.matmul(m, m_t)
    d = np.array([1.0 / x if np.abs(x) > 1.0e-8 else 0.0 for x in np.sum(p, axis=0)])
    return np.matmul(m_t, np.diag(d))

def gansynth_onnx_inference(model_path, midi_note, latent_vector, output_wav):
    print(f"Loading model from {model_path}...")
    sess = ort.InferenceSession(model_path)
    
    # MIDI 60 -> Label 36 (60 - 24)
    #For the NSynth dataset, MIDI 24 (C1) corresponds to index 0, and MIDI 60 (C4) corresponds to index `36`.
    MIN_MIDI_PITCH = 24
    label = midi_note - MIN_MIDI_PITCH
    label_input = np.array([label], dtype=np.int32)
    
    print(f"Generating MIDI {midi_note} (Internal Label: {label})...")
    outputs = sess.run(None, {'Placeholder:0': label_input, 'Placeholder_1:0': latent_vector})
    
    # [128, 1024, 2] (Time, Mel_Bins, Channels)
    specgram = outputs[0][0]
    logmelmag2 = specgram[:, :, 0]
    mel_ifreq = specgram[:, :, 1]
    
    print("Output shapes:")
    print("Log-Mel Magnitude shape:", logmelmag2.shape)
    print("Mel IFreq shape:", mel_ifreq.shape)  
    
    # Constants
    SR = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MEL = 1024
    N_MAG = 1025 # (N_FFT // 2) + 1
    
    # Preparation of the inverse transformation matrix (approximate inverse linear transformation matrix)
    mel2l = get_mel_to_linear_matrix(N_MEL, N_MAG, SR)
    
    # Export the mel2l matrix as a csv file for debugging
    np.savetxt('pythonScripts/mel2l_matrix.csv', mel2l, fmt='%.6f', delimiter=',')
    
    #print("mel2l shape:", mel2l.shape)  
    #print(mel2l[:5, :5])  # Print the top-left 5x5 block of the matrix for verification
    
    # 1. Magnitude Conversion: Mel LogPower -> Linear Magnitude
    mag2_linear = np.dot(np.exp(logmelmag2), mel2l)
    mag_linear = np.sqrt(np.maximum(0, mag2_linear))
    
    # 2. Phase transformation: Mel IFreq -> Mel Phase -> Linear Phase
    mel_phase = np.cumsum(mel_ifreq * np.pi, axis=0)
    linear_phase = np.dot(mel_phase, mel2l)
    
    # 3. ISTFT
    print("Performing ISTFT...")
    stft = mag_linear * np.exp(1j * linear_phase)
    print("STFT shape:", stft.shape)
    audio = librosa.istft(stft.T, hop_length=HOP_LENGTH, win_length=N_FFT)
    
    # 4. Save
    # Normalize audio to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    print("audio shape:", audio.shape)
    sf.write(output_wav, audio, SR)
    print(f"Success! Saved to {output_wav}")

if __name__ == "__main__":
    model_path = 'model/gansynth.onnx'
    latent_vector = np.random.normal(size=(1, 256)).astype(np.float32)
    
    #for midi_note in [60, 64, 67]:  # C4, E4, G4
    #    output_wav = f'pythonScripts/outputAudio/output_gansynth_{midi_note}.wav'
    #    if os.path.exists(model_path):
    #        gansynth_onnx_inference(model_path, midi_note, latent_vector, output_wav)
    #    else:
    #        print(f"File not found: {model_path}")
    
    # Example for MIDI 60 (C4)
    midi_note = 60
    output_wav = f'pythonScripts/outputAudio/output_gansynth_{midi_note}.wav'
    if os.path.exists(model_path):
        gansynth_onnx_inference(model_path, midi_note, latent_vector, output_wav)
    else:
        print(f"File not found: {model_path}")
