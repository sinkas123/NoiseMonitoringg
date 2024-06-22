import librosa
import numpy as np
import matplotlib.pyplot as plt

# Datei Pfad zur Audiodatei (ersetzen Sie dies durch den Pfad Ihrer Audiodatei)
audio_path = 'audio/file_example_WAV_2MG.wav'

# Audio laden
y, sr = librosa.load(audio_path)

# Mel-Spektrogramm berechnen
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# MFCCs berechnen
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Lautstärke berechnen
rms = librosa.feature.rms(y=y)

# Mel-Spektrogramm anzeigen
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.tight_layout()
plt.show()

# MFCCs anzeigen
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Lautstärke anzeigen
plt.figure(figsize=(10, 4))
plt.semilogy(rms.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms.shape[-1]])
plt.legend(loc='best')
plt.title('Root-Mean-Square (RMS) Energy')
plt.tight_layout()
plt.show()

# Ergebnisse ausgeben
print(f"Mel-Spectrogramm Shape: {mel_spectrogram.shape}")
print(f"MFCCs Shape: {mfccs.shape}")
print(f"Lautstärke (RMS) Shape: {rms.shape}")
