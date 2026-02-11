import librosa
import numpy as np

def to_mel(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel
