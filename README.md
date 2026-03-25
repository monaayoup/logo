---
title: Logo Generator
emoji: ??
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
# Create/update README with HF metadata

# TalentTree Logo Generator 🎨

An AI-powered logo generator that uses a conversational chatbot to collect brand information and automatically generates professional logos using Stable Diffusion XL.

---

## How It Works

1. The user chats with an AI bot that collects brand details (name, industry, audience, tone, style, colors)
2. When enough info is gathered and the user says "generate", the bot crafts an optimized image prompt
3. Stable Diffusion XL generates a professional logo based on that prompt
4. The logo is returned as a base64-encoded PNG

---

## Tech Stack

- **LLM** — `meta-llama/Llama-3.3-70B-Instruct` via Hugging Face Inference API
- **Image Model** — `stabilityai/stable-diffusion-xl-base-1.0` via Hugging Face Inference API
- **Backend** — FastAPI + Uvicorn
- **Containerization** — Docker

---

## Project Structure

```
TALENTREELOGO/
├── backend_api.py      # FastAPI app — all endpoints and logic
├── logo.py             # Original CLI version of the chatbot
├── Requirements.txt    # Python dependencies
├── Dockerfile          # Docker container config
├── .env                # Environment variables (not committed)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## Setup & Running

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd TALENTREELOGO
```

### 2. Create your `.env` file
```bash
HF_TOKEN=your_huggingface_token_here
```
Get your token from: https://huggingface.co/settings/tokens

### 3. Install dependencies
```bash
pip install -r Requirements.txt
```

### 4. Run locally
```bash
uvicorn backend_api:app --reload
```
API will be available at: `http://127.0.0.1:8000`

---

## Running with Docker

### Build the image
```bash
docker build -t talentree-logo .
```

### Run the container
```bash
docker run -p 8000:8000 --env-file .env talentree-logo
```
API will be available at: `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/`        | Health check                                     |
| GET    | `/opening` | Returns the bot's welcome message                |
| POST   | `/chat`    | Send a message, get bot reply + generation flag  |
| POST   | `/generate`| Generate logo from conversation history          |

### POST `/chat`
**Request:**
```json
{
  "message": "My brand is NovaBrew, a specialty coffee shop",
  "history": []
}
```
**Response:**
```json
{
  "reply": "Great! What style do you prefer for your logo?",
  "should_generate": false
}
```

### POST `/generate`
**Request:**
```json
{
  "history": [
    { "role": "user", "content": "My brand is NovaBrew..." },
    { "role": "assistant", "content": "What style do you prefer?" },
    { "role": "user", "content": "Minimalist, brown and cream, generate" },
    { "role": "assistant", "content": "Perfect! Generating your logo now..." }
  ]
}
```
**Response:**
```json
{
  "image_base64": "<base64-encoded PNG string>",
  "prompt_used": "A professional vector logo for NovaBrew...",
  "brand_name": "NovaBrew"
}
```

---

## Interactive API Docs

Once the server is running, visit:
```
http://127.0.0.1:8000/docs
```
This opens the Swagger UI where you can test all endpoints directly in the browser.

---

## Environment Variables

| Variable  | Description                        |
|-----------|------------------------------------|
| `HF_TOKEN`| Your Hugging Face API access token |

---

## .gitignore Recommendations

Make sure your `.gitignore` includes:
```
.env
__pycache__/
*.pyc
*.png
.venv/
```
