import librosa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

y, sr = librosa.load(librosa.ex("trumpet"))
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch = [
    np.max(pitches[:, i]) if np.max(magnitudes[:, i]) > 0 else 0
    for i in range(pitches.shape[1])
]


plt.figure(figsize=(14, 5))
plt.plot(pitch, label="Pitch")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch over Time")
plt.legend()
plt.show()
