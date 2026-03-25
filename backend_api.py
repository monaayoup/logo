import os
import io
import base64
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LLM_MODEL   = "meta-llama/Llama-3.3-70B-Instruct"

app = FastAPI(title="Logo Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert brand identity designer and AI image prompt engineer.
Your ONLY job is to help users generate high-quality logos for their brand.

When gathering brand information, ask about:
- Brand name
- Industry / niche
- Target audience
- Style preferences (minimalist, bold, playful, luxury, techy, organic, etc.)
- Color preferences
- Any symbols or icons they have in mind

When you have enough information and are asked to generate, output a SINGLE image generation prompt.

A great logo prompt follows this exact structure:
"A professional vector logo for [BRAND NAME], [ICON DESCRIPTION], [STYLE ADJECTIVES] style, [COLOR PALETTE], [MOOD/FEEL], isolated on white background, no text, clean crisp edges, SVG-quality, award-winning logo design, Dribbble trending"

Rules for the prompt:
- Be highly specific about the icon shape (e.g. "a geometric fox formed by triangular facets" not just "a fox")
- Name exact colors (e.g. "deep navy #1B2A4A and electric teal #00C2CC" not "blue and green")
- Specify the style clearly (flat design / line art / monogram / lettermark / geometric / emblematic / mascot)
- Avoid gradients and shadows — logos must be flat and clean
- NEVER include raster-style descriptions (no "photorealistic", no "3D render", no "bokeh")

Only discuss logo design. If asked about anything else, redirect the user back to logo creation."""

OPENING_MESSAGE = """👋 Welcome to the AI Logo Generator!

I'll design a professional logo for your brand. To get started, please tell me:

🏷️  Brand Name — What's your brand called?
🏭  Industry / Niche — What does your brand do?
👥  Target Audience — Who are your customers?
🎭  Brand Tone — Bold, elegant, playful, minimal?
🎨  Style Preference — Flat icon, geometric, monogram, mascot?
🖌️  Color Preferences — Any colors or moods in mind?
💡  Symbol / Icon — Any specific icon idea? (optional)

Answer all at once or one by one — when you're ready say 'generate'!"""

# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"/"assistant", "content": "..."}]

class ChatResponse(BaseModel):
    reply: str
    should_generate: bool

class GenerateRequest(BaseModel):
    history: list[dict]        # Full conversation history

class GenerateResponse(BaseModel):
    image_base64: str          # PNG encoded as base64
    prompt_used: str
    brand_name: str

# ── Helper functions ──────────────────────────────────────────────────────────
def build_messages(history: list[dict]) -> list[dict]:
    """Prepend system prompt to conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def get_bot_reply(history: list[dict], user_message: str) -> str:
    """Get a conversational reply from the LLM."""
    messages = build_messages(history + [{"role": "user", "content": user_message}])
    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.7
    )
    return resp.choices[0].message["content"].strip()


def check_trigger(user_message: str) -> bool:
    """Check if the user's message contains a generation trigger word."""
    TRIGGER_WORDS = {
        "generate", "create", "make", "yes", "go ahead", "ready",
        "logo now", "let's go", "do it", "build it", "make it"
    }
    return any(word in user_message.lower() for word in TRIGGER_WORDS)


def bot_still_asking(bot_reply: str) -> bool:
    """Check if the bot is still waiting for more info."""
    QUESTION_INDICATORS = [
        "can you please", "could you", "what is", "what are",
        "please provide", "i'd like to", "i need", "tell me"
    ]
    return any(phrase in bot_reply.lower() for phrase in QUESTION_INDICATORS)


def extract_brand_name(history: list[dict]) -> str:
    """Extract the brand name from conversation history."""
    messages = build_messages(history) + [{
        "role": "user",
        "content": "What is the brand name from our conversation? Reply with ONLY the brand name, nothing else."
    }]
    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=20,
        temperature=0.1
    )
    return resp.choices[0].message["content"].strip()


def build_logo_prompt(history: list[dict]) -> str:
    """Build an optimized image generation prompt from conversation history."""
    messages = build_messages(history) + [{
        "role": "user",
        "content": (
            "Now write the final image generation prompt for this logo following the exact structure "
            "described in your instructions. Output ONLY the prompt, nothing else — no explanation, "
            "no preamble, no quotes around it."
        )
    }]
    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=200,
        temperature=0.6
    )
    return resp.choices[0].message["content"].strip()


def run_image_generation(prompt: str) -> bytes:
    """Run SDXL image generation and return raw PNG bytes."""
    final_prompt = (
        f"{prompt}, "
        "vector logo design, flat design, professional branding, "
        "white background, no shadows, no gradients, no noise, "
        "crisp clean lines, high resolution, sharp edges, "
        "no text, no letters, no watermark, no drop shadow, no blur, "
        "no realistic photo, no 3D render, no gradient background"
    )
    image = client.text_to_image(final_prompt, model=IMAGE_MODEL)

    # Convert PIL image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Logo Generator API is running"}


@app.get("/opening")
def opening():
    """Return the bot's opening welcome message."""
    return {"message": OPENING_MESSAGE}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a user message and get the bot's reply.
    Also returns should_generate=True when the bot has enough info
    and the user used a trigger word — frontend should then call /generate.
    """
    try:
        reply = get_bot_reply(req.history, req.message)

        # Decide whether the frontend should now call /generate
        triggered = check_trigger(req.message)
        still_asking = bot_still_asking(reply)
        should_generate = triggered and not still_asking

        return ChatResponse(reply=reply, should_generate=should_generate)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Given the full conversation history, generate a logo.
    Returns the image as a base64-encoded PNG string.
    """
    try:
        # Extract brand name
        brand_name = extract_brand_name(req.history)
        if not brand_name or brand_name.lower() in {"none", "unknown", "n/a", ""}:
            raise HTTPException(status_code=400, detail="Brand name could not be detected from conversation.")

        # Build the prompt
        logo_prompt = build_logo_prompt(req.history)

        # Generate the image
        image_bytes = run_image_generation(logo_prompt)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return GenerateResponse(
            image_base64=image_base64,
            prompt_used=logo_prompt,
            brand_name=brand_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))