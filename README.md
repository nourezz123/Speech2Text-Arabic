# 🎙️ Deep Learning Based Arabic Audio Understanding and Retrieval System

> An end-to-end Automatic Speech Recognition (ASR) pipeline for Arabic language using deep learning, built with OpenAI Whisper and evaluated on two Arabic speech datasets.

---

## 📌 Project Overview

This project implements an Arabic Speech Recognition system capable of transcribing Arabic audio into text. The system leverages **OpenAI Whisper Medium**, a state-of-the-art transformer-based model trained on 680,000 hours of multilingual speech data, with native Arabic support.

The project is structured into two Jupyter notebooks:
- **`eda.ipynb`** — Exploratory Data Analysis of both datasets
- **`main.ipynb`** — Full ASR pipeline, evaluation, and interactive demo

---

## 🏗️ System Architecture

```
Audio Input (.wav / .mp3)
        │
        ▼
Preprocessing
(Resampling → 16kHz, Normalization)
        │
        ▼
Feature Extraction
(Log-Mel Spectrogram — 80 bins, 25ms window)
        │
        ▼
Whisper Medium Model
(CNN Stem → Transformer Encoder → Transformer Decoder)
        │
        ▼
Arabic Text Output (Unicode)
        │
        ▼
Evaluation (WER / CER)
```

### Why CNN + Transformer?
| Component | Role |
|-----------|------|
| **CNN Stem** | Extracts local acoustic features from the spectrogram |
| **Transformer Encoder** | Captures long-range dependencies across the audio |
| **Transformer Decoder** | Auto-regressively generates Arabic text tokens |

---

## 📁 Datasets

### Dataset 1 — Arabic Speech Corpus
| Property | Value |
|----------|-------|
| Source | [en.arabicspeechcorpus.com](https://en.arabicspeechcorpus.com) |
| Speaker | Single native MSA speaker |
| Quality | Studio-recorded (high SNR) |
| Format | WAV, 16kHz |
| Samples | 100 test sentences |
| References | Buckwalter phonetic transliteration |

### Dataset 2 — Mozilla Common Voice Arabic
| Property | Value |
|----------|-------|
| Source | [Mozilla Common Voice](https://commonvoice.mozilla.org) |
| Speakers | Multiple (crowd-sourced) |
| Quality | Variable (consumer devices) |
| Format | MP3, 48kHz |
| References | Arabic script ✅ (enables real WER) |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Word Error Rate (WER)** | Reported in `main.ipynb` |
| **Character Error Rate (CER)** | Reported in `main.ipynb` |
| **Word Accuracy** | Reported in `main.ipynb` |
| **Dataset 1 Quality** | Visually accurate (phonetic refs only) |


---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- FFmpeg (required by Whisper for audio decoding)

### Install FFmpeg (Windows)
1. Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract and move to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH
4. Verify: `ffmpeg -version`

### Install Python Dependencies

```bash
pip install openai-whisper torch torchaudio datasets transformers jiwer gradio soundfile matplotlib seaborn librosa arabic-reshaper python-bidi
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Exploratory Data Analysis
Open and run `eda.ipynb` to explore the datasets:
```bash
jupyter notebook eda.ipynb
```

### 2. Run the Full Pipeline
Open and run `main.ipynb` to transcribe audio and evaluate:
```bash
jupyter notebook main.ipynb
```

### 3. Interactive Demo
The last cell of `main.ipynb` launches a Gradio web interface:
- Upload any Arabic `.wav` or `.mp3` file
- Or record directly from your microphone
- Get instant Arabic text transcription

---

## 📦 Requirements

```
openai-whisper
torch
torchaudio
datasets
transformers
jiwer
gradio
soundfile
matplotlib
seaborn
librosa
arabic-reshaper
python-bidi
pandas
numpy
```

---

## 🔍 Notebooks Overview

### `eda.ipynb`
- Audio waveform visualization
- Log-Mel spectrogram analysis
- MFCC feature extraction
- Duration & word count distributions
- Dataset comparison (ASC vs Common Voice)
- Signal quality analysis (SNR, RMS)
- Word frequency analysis

### `main.ipynb`
- Whisper model family comparison
- Full transcription of Arabic Speech Corpus (100 files)
- Full transcription of Common Voice Arabic (100 samples)
- WER & CER evaluation with visualizations
- Best/worst prediction analysis
- Gradio interactive demo interface

---


## 📚 References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., OpenAI (2022)
- [Arabic Speech Corpus](https://en.arabicspeechcorpus.com) — Halabi (2016)
- [Mozilla Common Voice](https://commonvoice.mozilla.org) — Mozilla Foundation
- [jiwer — WER computation](https://github.com/jitsi/jiwer)

---

## 👤 Author

**Nour Ezz** 
