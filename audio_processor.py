"""
audio_processor.py — VocalCanvas Unified Audio Processor
=========================================================
Single source of truth for all DSP operations.
Import this in BOTH preprocess.py and predict.py.

Public API
----------
chunk_to_spectrogram(chunk)
    Raw audio chunk  →  (IMG_SIZE, IMG_SIZE, 1) float32 [0,1]   ← CNN input

is_silent(chunk)         → bool
split_audio(audio)       → list[np.ndarray]
"""

import librosa
import numpy as np
from PIL import Image


class UnifiedAudioProcessor:
    """
    Encapsulates every DSP step that the model depends on.
    All parameters that affect spectrogram shape or value range live here.

    Training contract
    -----------------
    - 3-second chunks at 22 050 Hz  →  66 150 samples
    - Mel spectrogram: 128 bands, librosa defaults for n_fft / hop_length / window
    - power_to_db with ref=np.max
    - Local min-max normalization to [0, 1] in float32  ← pure numpy, no matplotlib
    - PIL resize to (IMG_SIZE, IMG_SIZE) with BILINEAR  ← same resampler both ways
    - Output shape: (IMG_SIZE, IMG_SIZE, 1)
    - Silence threshold for skipping chunks: MAX_ABS_THRESHOLD

    Never amplitude-normalize the audio before calling chunk_to_spectrogram().
    """

    # ------------------------------------------------------------------
    # DSP parameters — change these in ONE place only
    # ------------------------------------------------------------------
    SAMPLE_RATE: int      = 22_050
    CHUNK_DURATION: int   = 3           # seconds
    N_MELS: int           = 128
    # n_fft, hop_length, window: librosa defaults (2048, 512, hann)

    IMG_SIZE: int         = 128         # square output image side length

    # Chunks whose peak amplitude is below this are considered silence
    SILENCE_THRESHOLD: float = 0.01

    # ------------------------------------------------------------------
    # Derived constants
    # ------------------------------------------------------------------
    @property
    def chunk_samples(self) -> int:
        """Exact number of samples in one chunk (integer, no rounding needed)."""
        return self.CHUNK_DURATION * self.SAMPLE_RATE   # 66 150

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_silent(self, chunk: np.ndarray) -> bool:
        """Return True if the chunk should be skipped as silence."""
        return np.max(np.abs(chunk)) < self.SILENCE_THRESHOLD

    def chunk_to_spectrogram(self, chunk: np.ndarray) -> np.ndarray:
        """
        Convert a 1-D audio chunk (float32, chunk_samples long) to a
        normalized spectrogram ready for model input.

        Returns
        -------
        np.ndarray  shape (IMG_SIZE, IMG_SIZE, 1), dtype float32, range [0, 1]
        """
        # 1. Mel spectrogram: This converts audio signals into a visual representation
        # where the y-axis is the frequency and x-axis is time. 
        # 'n_mels' controls the number of frequency bands.
        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=self.SAMPLE_RATE,
            n_mels=self.N_MELS,
        )

        # 2. Power → dB: Neural networks prefer numbers that aren't extremely large or small.
        # Converting power to decibels (dB) puts the data into a logarithmic scale, 
        # which closely mimics how human ears perceive loudness.
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 3. Local min-max normalization: Squish the data to perfectly fit between 0.0 and 1.0.
        # This makes training the CNN (Convolutional Neural Network) much faster and more stable.
        denom = mel_db.max() - mel_db.min()
        mel_norm = (mel_db - mel_db.min()) / (denom + 1e-9)   # The 1e-9 prevents division by zero!

        # 4. Resize the image: The CNN expects a strictly squared image (e.g., 128x128).
        # We use the PIL library to resize our spectrogram perfectly.
        pil_img = Image.fromarray(mel_norm.astype(np.float32), mode='F')
        pil_img = pil_img.resize(
            (self.IMG_SIZE, self.IMG_SIZE),
            resample=Image.BILINEAR,
        )

        # 5. Prepare for CNN: Convert back to a numpy array and add an extra "channel" dimension.
        # Just like colored images have RGB channels, our grayscale spectrogram has 1 channel.
        img_array = np.array(pil_img, dtype=np.float32)        # Shape: (128, 128)
        img_array = np.expand_dims(img_array, axis=-1)         # Shape: (128, 128, 1)

        return img_array

    def split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Split a 1-D audio array into non-overlapping chunks.
        The final partial chunk is discarded to ensure every chunk is exactly the same size.

        Parameters
        ----------
        audio : np.ndarray   shape (N,), dtype float32

        Returns
        -------
        list of np.ndarray, each of length chunk_samples
        """
        n = self.chunk_samples
        # Slice the audio from start to end, taking steps of 'n' (the chunk size)
        return [
            audio[i : i + n]
            for i in range(0, len(audio) - n, n)
        ]
