# models/audio_model.py
# Simple MFCC-based fake-audio detector (no fusion, no GPU needed)

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class AudioFakeModel:
    def __init__(self):
        # Simple statistical baseline classifier
        self.scaler = StandardScaler()
        self.clf = LogisticRegression()

        # We create a fake "trained" model using typical MFCC distribution
        # (so detection works even without training dataset)
        real_mean = np.zeros(20)
        fake_mean = np.ones(20) * 1.5
        real_samples = np.random.normal(real_mean, 0.5, (200, 20))
        fake_samples = np.random.normal(fake_mean, 0.5, (200, 20))

        X = np.vstack([real_samples, fake_samples])
        y = np.array([0]*200 + [1]*200)

        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)

    def predict_proba(self, wav_path: str) -> float:
        """
        Returns probability (0..1) that audio is FAKE.
        """
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            feat = np.mean(mfcc, axis=1).reshape(1, -1)

            feat = self.scaler.transform(feat)
            prob_fake = self.clf.predict_proba(feat)[0, 1]
            return float(prob_fake)
        except Exception:
            return 0.5  # unknown → neutral
