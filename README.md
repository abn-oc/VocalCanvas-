# VocalCanvas 🎙️

VocalCanvas is a Deep Learning-based speaker identification system. It uses a Convolutional Neural Network (CNN) trained on Mel Spectrogram images to accurately identify speakers from raw audio files.

## Setup Instructions

**1. Install Prerequisites**
- **Python 3.8+**
- **FFmpeg** (Required for audio conversion)
  - *Windows*: `winget install ffmpeg`
  - *Mac*: `brew install ffmpeg`
  - *Linux*: `sudo apt install ffmpeg`

**2. Virtual Environment & Dependencies**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*(No additional environment variables are required!)*

## Usage

**1. Train the Model**
You must train the CNN model first before making predictions:
```bash
python train.py
```

**2. Command Line Prediction**
Test the model on an audio file directly from the terminal:
```bash
python predict.py path/to/audio.wav
```

**3. Streamlit Dashboard**
Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```
