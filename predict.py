"""
predict.py — VocalCanvas Inference
==============================================
Usage:
    python predict.py <audio_file>

This uses the trained CNN model to predict the speaker.
"""

import argparse
import librosa
import numpy as np
import os
import subprocess
import sys

import tensorflow as tf

from audio_processor import UnifiedAudioProcessor

PROC = UnifiedAudioProcessor()

# =========================================================
# Config — must match train.py exactly
# =========================================================
CNN_MODEL_PATH  = "models/vocalcanvas_cnn.keras"   # written by train.py Section 1

SPEAKERS = [
    "abi",
    "ahmed",
    "zoha"
]

# DSP constants from the single source of truth
SAMPLE_RATE    = PROC.SAMPLE_RATE
CHUNK_DURATION = PROC.CHUNK_DURATION
IMG_SIZE       = PROC.IMG_SIZE

# =========================================================
# Shared utility: convert any audio file to wav
# =========================================================
def convert_to_wav(input_path: str) -> str:

    base = os.path.splitext(input_path)[0]
    output_path = base + "_converted.wav"

    print("Converting to wav...")

    subprocess.run(
        [
            "ffmpeg",
            "-i",  input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            output_path,
            "-y"
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output_path

# =========================================================
# Shared utility: load + chunk audio
# =========================================================
def load_and_chunk(audio_path: str):
    """
    Returns
    -------
    chunks : list[np.ndarray]  each of shape (chunk_samples,)
    """
    ext = os.path.splitext(audio_path)[1].lower()

    if ext != ".wav":
        audio_path = convert_to_wav(audio_path)

    print(f"\nLoading audio: {audio_path}")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    # No amplitude boost — training data was not boosted.
    # Quiet audio is handled naturally by per-chunk min-max
    # normalization inside UnifiedAudioProcessor.

    chunks = PROC.split_audio(audio)

    print(f"Total chunks: {len(chunks)}")

    return chunks

# =========================================================
# Shared utility: majority-vote result printer
# =========================================================
def print_results(votes: dict, confidences: list):

    print("\n--- Results ---")

    total = sum(votes.values())

    if total == 0:
        print("No valid (non-silent) chunks detected.")
        return

    for speaker, count in votes.items():
        print(
            f"  {speaker}: "
            f"{count}/{total} chunks "
            f"({count / total * 100:.1f}%)"
        )

    winner = max(votes, key=votes.get)

    print(f"\n  ▶  Predicted speaker : {winner.upper()}")

    if confidences:
        avg_conf = np.mean(confidences) * 100
        print(f"     Average confidence: {avg_conf:.1f}%")

# =========================================================
# ███████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
# ███████╗█████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
# ╚════██║██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
# ███████║███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
# ╚══════╝╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
# CNN prediction path
# =========================================================
def load_cnn_model():
    print(f"Loading CNN model from {CNN_MODEL_PATH} ...")
    return tf.keras.models.load_model(CNN_MODEL_PATH)


def predict_cnn(audio_path: str, model) -> None:
    """
    Run CNN inference on the audio file.
    The audio is broken into short chunks. Each chunk is converted to a spectrogram,
    and the model predicts who is speaking in that chunk.
    The final answer is a majority vote across all valid chunks.
    """
    # 1. Load the audio file and split it into chunks
    chunks = load_and_chunk(audio_path)

    # Keep track of how many chunks were predicted for each speaker
    votes       = {s: 0 for s in SPEAKERS}
    confidences = []

    for idx, chunk in enumerate(chunks):

        # Skip chunks that are completely silent
        if PROC.is_silent(chunk):
            print(f"  Chunk {idx:03d}: SKIPPED (silence)")
            continue

        # 2. Convert the raw audio chunk into a spectrogram image
        spec = PROC.chunk_to_spectrogram(chunk)
        # 3. Add an extra dimension to represent a "batch" of 1 image
        spec = np.expand_dims(spec, axis=0)

        # 4. Have the model guess who it is! It returns probabilities for each speaker.
        probs      = model.predict(spec, verbose=0)[0]
        # Find the index of the highest probability
        pred_idx   = int(np.argmax(probs))
        # This is how confident the model is (e.g. 0.95 = 95%)
        confidence = float(probs[pred_idx])
        # Get the actual speaker's name
        speaker    = SPEAKERS[pred_idx]

        # 5. Record the vote
        votes[speaker] += 1
        confidences.append(confidence)

        print(
            f"  Chunk {idx:03d}: "
            f"{speaker} "
            f"({confidence * 100:.1f}%)"
        )

    # 6. Show the final results
    print_results(votes, confidences)





# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="VocalCanvas — Speaker Identification"
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to identify."
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Validate input file
    # ----------------------------------------------------------
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found — {args.audio_file}")
        sys.exit(1)

    # ----------------------------------------------------------
    # Dispatch to the CNN model
    # ----------------------------------------------------------
    print(f"\nModel type : CNN (Convolutional Neural Network)")
    print(f"Audio file : {args.audio_file}")
    print("-" * 50)

    # Load the trained model from our file
    model = load_cnn_model()
    # Run the prediction
    predict_cnn(args.audio_file, model)