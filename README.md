# Call QA: Automated Quality Scoring for Indian Contact Centres

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-f55036.svg)](https://console.groq.com)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-blue.svg)](https://github.com/SYSTRAN/faster-whisper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Every BPO, insurance operation, fintech, and D2C brand in India records 100% of calls — and listens to roughly 2%.**  
> **Call QA scores 100% of calls automatically using open-source Whisper, speaker diarization, Groq Llama 3.3 70B, and Streamlit.**

Instead of QA managers randomly sampling 50 calls, the Call QA dashboard prioritizes the **top 20 worst/flagged calls** (compliance breaches, customer escalations, prohibited words) for immediate intervention.

---

## 🏗️ Architecture Pipeline

```text
Audio File (.mp3 / .wav)
      │
      ▼
faster-whisper (local/cloud)  ──►  Raw transcript + timestamps (Hindi / English / Hinglish)
      │
      ▼
pyannote-audio 3.1            ──►  Speaker diarization (Agent vs. Customer separation)
      │
      ▼
Segment Merge                 ──►  [00:12 Agent] "Good morning, thank you for calling..."
      │
      ▼
Groq LLM (Llama 3.3 70B)      ──►  10-Dimension Rubric (0-10) + 3 Risk Flags (Structured JSON)
      │
      ▼
SQLite / Supabase Postgres    ──►  Stored audit records & aggregate metrics
      │
      ▼
Streamlit Dashboard           ──►  Executive KPIs + Drilldown Transcript + CSV Export
```

---

## 📋 Starter 10-Dimension Rubric

| # | Dimension | Reviewer Focus |
|---|-----------|----------------|
| 1 | **Greeting & Brand Identity** | Proper greeting, agent name, and clear brand attribution? |
| 2 | **Customer Verification** | Identity authentication (mobile, order ID, DOB) before sharing account data? |
| 3 | **Active Listening** | Attentive listening without interrupting or talking over the customer? |
| 4 | **Empathy & Tone** | Sincere empathy and understanding of customer distress? |
| 5 | **Clarity of Communication** | Jargon-free, clear explanations, appropriate Hindi/English language match? |
| 6 | **Problem Resolution** | Query resolved or clear ticket reference and TAT provided? |
| 7 | **Compliance Disclosures** | Mandatory regulatory/legal disclosures read (e.g., call recording)? |
| 8 | **Hold / Mute Handling** | Customer permission asked before hold, time estimated, thanked upon return? |
| 9 | **Objection Handling** | Constructive and composed de-escalation when customer pushed back? |
| 10 | **Closing** | Next steps summarized, additional help checked, warm brand sign-off? |

### 🚨 Critical Categorical Flags (Boolean)
- **Compliance Violation**: Any mandatory regulatory, KYC, or statutory disclaimer missed.
- **Customer Escalated**: Shouting, abusive language, or supervisor/ombudsman demanded.
- **Prohibited Words**: Slang, unauthorized promises (*"pakka guarantee refund"*), or competitor mentions.

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/kndkrakhil/CallQA.git
cd CallQA

# Install requirements
pip install -r requirements.txt
```

> **FFmpeg Note:** Ensure `ffmpeg` is installed on your operating system for audio conversion:
> - **Ubuntu/Debian**: `sudo apt install ffmpeg`
> - **macOS**: `brew install ffmpeg`
> - **Windows**: `winget install ffmpeg`

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your free API keys:
```bash
cp .env.example .env
```
Key configuration values:
- `GROQ_API_KEY`: Get a free key at [console.groq.com](https://console.groq.com) (14,400 free requests/day).
- `GEMINI_API_KEY`: (Optional backup) Get at [aistudio.google.com](https://aistudio.google.com).
- `HF_TOKEN`: (Optional) Hugging Face token for `pyannote/speaker-diarization-3.1`.

*Note: If no API key is provided, the system automatically runs with an offline heuristic rule engine so you can test immediately.*

### 3. Generate Sample Call & Run Pipeline (CLI)
```bash
# Generate sample audio & companion transcript
python samples/generate_sample.py

# Run end-to-end audit on sample call
python run.py samples/sample_call.wav --save-db
```

### 4. Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Features

1. **Batch Upload**: Drag-and-drop multiple MP3 or WAV call recordings.
2. **Flagged Calls Priority List**: Top 20 worst calls ranked by compliance violation severity and low scores.
3. **Deep-Dive Transcript**: Color-coded chat bubbles (`Agent` vs `Customer`) with exact timestamps and inline score rationale.
4. **1-Click CSV Export**: Download complete QA reports with every call and score dimension.

---

## 🚢 Deployment Guide

### Option 1: Streamlit Community Cloud (Free & Instant)
1. Fork or push this repository to your GitHub account (`kndkrakhil/CallQA`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository `kndkrakhil/CallQA`, Branch `main`, and Main file path `app.py`.
4. In **Advanced Settings**, paste your `GROQ_API_KEY` under Secrets:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **Deploy**!

### Option 2: Hugging Face Spaces (Free CPU Docker)
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Docker** SDK (Blank).
3. Push this repository or connect GitHub repo. The included `Dockerfile` will automatically install `ffmpeg`, dependencies, and boot `app.py`.
4. Add `GROQ_API_KEY` in Space **Settings -> Variables and Secrets**.

### Option 3: Oracle Cloud Always Free VM / Docker
The included `Dockerfile` is pre-configured with `ffmpeg` and system audio libraries:
```bash
# Build Docker image
docker build -t callqa:latest .

# Run container
docker run -d -p 8501:8501 --env-file .env --name callqa-app callqa:latest
```

---

## 🧪 Automated Testing

Run the test suite to verify rubric definitions, transcription formatting, scoring schemas, and database persistence:
```bash
python tests/test_pipeline.py
```

---

## 📄 License
MIT License. Built for Indian contact centres and modern BPO operations.
