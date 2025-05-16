# _Speech_to_Text_for_transcription_services.ipynb

This project notebook demonstrates the process of converting speech audio into text using modern speech recognition tools, alongside basic data cleaning, visualization, and error analysis.

## 🛠️ Setup and Installation

Installs necessary Python libraries:

* `kaggle`, `transformers`, `torchaudio`, `librosa`, `noisereduce` – for audio handling and ML.
* `openai-whisper` – optional, for OpenAI's Whisper transcription model.
* `datasets` – for handling dataset structures.
* `jiwer` – for computing Word Error Rate (WER).

```bash
!pip install kaggle
!pip install transformers torchaudio librosa noisereduce
!pip install openai-whisper
!pip install datasets jiwer
```

## 📤 Data Upload

Uploads dataset via manual file selection in Google Colab:

```python
from google.colab import files
files.upload()  # Upload a .tar.gz audio dataset
```

## 🔊 Audio Preprocessing

### Steps:

* Load an audio file using `librosa`.
* Trim silence from the beginning and end.
* Normalize volume levels.
* Visualize the cleaned waveform.

```python
y, sr = librosa.load(audio_path, sr=None)
y_clean, _ = librosa.effects.trim(y)
y_normalized = librosa.util.normalize(y_clean)
```

## 📈 Spectrogram Visualization

Displays a log-scaled spectrogram to inspect audio features:

```python
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_normalized)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
```

## 📊 Word Error Rate (WER) Analysis

Uses the `jiwer` package to compare actual vs. predicted transcriptions.

### Example:

```python
from jiwer import wer

actual_transcription = "your actual transcription here"
predicted_transcription = "your predicted transcription here"

wer_score = wer(actual_transcription, predicted_transcription)
print(f"Word Error Rate (WER): {wer_score}")
```

### Visualization:

Plots WER scores for multiple test cases using `matplotlib`.


