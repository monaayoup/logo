import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import gradio as gr

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LLM_MODEL   = "meta-llama/Llama-3.3-70B-Instruct"

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert brand identity designer and AI image prompt engineer.
Your ONLY job is to help users generate high-quality logos for their brand.

When you have enough information, output a SINGLE image generation prompt.

A great logo prompt follows this exact structure:
"A professional vector logo for [BRAND NAME], [ICON DESCRIPTION], [STYLE ADJECTIVES] style, [COLOR PALETTE], [MOOD/FEEL], isolated on white background, no text, clean crisp edges, SVG-quality, award-winning logo design, Dribbble trending"

Rules for the prompt:
- Be highly specific about the icon shape (e.g. "a geometric fox formed by triangular facets" not just "a fox")
- Name exact colors (e.g. "deep navy #1B2A4A and electric teal #00C2CC" not "blue and green")
- Specify the style clearly (flat design / line art / monogram / lettermark / geometric / emblematic / mascot)
- Avoid gradients and shadows — logos must be flat and clean
- NEVER include raster-style descriptions (no "photorealistic", no "3D render", no "bokeh")

Only discuss logo design. If asked about anything else, redirect the user back to logo creation."""

TRIGGER_WORDS = {
    "generate", "create", "make", "yes", "go ahead", "ready",
    "logo now", "let's go", "do it", "build it", "make it"
}

QUESTION_INDICATORS = [
    "can you please", "could you", "what is", "what are",
    "please provide", "i'd like to", "i need", "tell me"
]

# ── Core functions ────────────────────────────────────────────────────────────
def build_messages(history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def extract_brand_name(history: list[dict]) -> str:
    messages = build_messages(history) + [{
        "role": "user",
        "content": "What is the brand name from our conversation? Reply with ONLY the brand name, nothing else."
    }]
    resp = client.chat_completion(model=LLM_MODEL, messages=messages, max_tokens=20, temperature=0.1)
    return resp.choices[0].message["content"].strip()


def build_logo_prompt(history: list[dict]) -> str:
    messages = build_messages(history) + [{
        "role": "user",
        "content": (
            "Now write the final image generation prompt for this logo following the exact structure "
            "described in your instructions. Output ONLY the prompt, nothing else — no explanation, "
            "no preamble, no quotes around it."
        )
    }]
    resp = client.chat_completion(model=LLM_MODEL, messages=messages, max_tokens=200, temperature=0.6)
    return resp.choices[0].message["content"].strip()


def generate_logo_image(history: list[dict]) -> Image.Image:
    prompt = build_logo_prompt(history)
    final_prompt = (
        f"{prompt}, vector logo design, flat design, professional branding, "
        "white background, no shadows, no gradients, no noise, crisp clean lines, "
        "high resolution, sharp edges, no text, no letters, no watermark, "
        "no drop shadow, no blur, no realistic photo, no 3D render"
    )
    image = client.text_to_image(final_prompt, model=IMAGE_MODEL)
    return image


def form_to_history(brand_name, industry, audience, tone, style, colors, symbol):
    """Convert form fields into a single user message and a fake assistant ack."""
    parts = [
        f"Brand Name: {brand_name}",
        f"Industry: {industry}",
        f"Target Audience: {audience}",
        f"Brand Tone: {tone}",
        f"Style Preference: {style}",
        f"Color Preferences: {colors}",
    ]
    if symbol.strip():
        parts.append(f"Symbol / Icon Idea: {symbol}")

    user_msg = "Here is my brand profile:\n" + "\n".join(parts)
    assistant_ack = (
        f"Got it! Here's what I have for **{brand_name}**:\n"
        + "\n".join(f"- {p}" for p in parts)
        + "\n\nGenerating your logo now... 🎨"
    )
    return user_msg, assistant_ack


# ── Gradio handlers ───────────────────────────────────────────────────────────
def handle_form_generate(brand_name, industry, audience, tone, style, colors, symbol, chat_history):
    """Called when user clicks Generate from the form."""
    if not brand_name.strip():
        chat_history.append({"role": "assistant", "content": "⚠️ Please enter a brand name before generating."})
        return chat_history, chat_history

    user_msg, assistant_ack = form_to_history(brand_name, industry, audience, tone, style, colors, symbol)

    # Add to chat history
    chat_history.append({"role": "user", "content": user_msg})
    chat_history.append({"role": "assistant", "content": assistant_ack})

    # Generate logo
    try:
        image = generate_logo_image(chat_history)

        # Convert PIL image to base64 to embed in chat
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        img_md = f"![logo](data:image/png;base64,{b64})"

        brand = extract_brand_name(chat_history)
        chat_history.append({
            "role": "assistant",
            "content": f"✅ Here's your logo for **{brand}**!\n\n{img_md}"
        })
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Generation failed: {e}"})

    return chat_history, chat_history


def handle_chat(user_message, chat_history):
    """Called when user sends a message in the chat."""
    if not user_message.strip():
        return "", chat_history, chat_history

    chat_history.append({"role": "user", "content": user_message})

    try:
        # Get bot reply
        resp = client.chat_completion(
            model=LLM_MODEL,
            messages=build_messages(chat_history),
            max_tokens=400,
            temperature=0.7
        )
        bot_reply = resp.choices[0].message["content"].strip()
        chat_history.append({"role": "assistant", "content": bot_reply})

        # Check if we should generate
        triggered = any(word in user_message.lower() for word in TRIGGER_WORDS)
        still_asking = any(phrase in bot_reply.lower() for phrase in QUESTION_INDICATORS)

        if triggered and not still_asking:
            chat_history.append({"role": "assistant", "content": "⏳ Generating your logo now..."})

            image = generate_logo_image(chat_history)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            img_md = f"![logo](data:image/png;base64,{b64})"

            brand = extract_brand_name(chat_history)
            chat_history.append({
                "role": "assistant",
                "content": f"✅ Here's your logo for **{brand}**!\n\n{img_md}"
            })

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Error: {e}"})

    return "", chat_history, chat_history


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="AI Logo Generator") as demo:

    # Shared conversation state
    history_state = gr.State([])

    gr.Markdown("""
    # 🎨 AI Logo Generator
    Fill in your brand details on the left and generate instantly,
    or chat with the bot on the right to refine and generate.
    """)

    with gr.Row():

        # ── LEFT: Brand Profile Form ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Brand Profile")

            brand_name = gr.Textbox(label="Brand Name", placeholder="e.g. NovaBrew")
            industry   = gr.Textbox(label="Industry / Niche", placeholder="e.g. Specialty coffee shop")
            audience   = gr.Textbox(label="Target Audience", placeholder="e.g. Young professionals, 25–35")
            tone       = gr.Dropdown(
                label="Brand Tone",
                choices=["Bold & Confident", "Warm & Friendly", "Elegant & Luxury",
                         "Playful & Fun", "Minimal & Clean", "Techy & Modern", "Organic & Natural"],
                allow_custom_value=True
            )
            style      = gr.Dropdown(
                label="Logo Style",
                choices=["Flat Icon", "Geometric", "Monogram / Lettermark",
                         "Line Art", "Mascot", "Emblem", "Abstract"],
                allow_custom_value=True
            )
            colors     = gr.Textbox(label="Color Preferences", placeholder="e.g. Earthy brown and cream")
            symbol     = gr.Textbox(label="Symbol / Icon Idea (optional)", placeholder="e.g. Coffee bean, mountain, star")

            generate_btn = gr.Button("🚀 Generate Logo", variant="primary", size="lg")

        # ── RIGHT: Chat Interface ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Chat with the Bot")

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": (
                    "👋 Hi! I'm your AI logo designer.\n\n"
                    "You can fill in the **Brand Profile** on the left and hit **Generate Logo**, "
                    "or just tell me about your brand here and say **'generate'** when ready!"
                )}],
                type="messages",
                height=500,
                show_label=False,
                render_markdown=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Tell me about your brand or say 'generate'...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    # ── Event handlers ────────────────────────────────────────────────────────
    generate_btn.click(
        fn=handle_form_generate,
        inputs=[brand_name, industry, audience, tone, style, colors, symbol, history_state],
        outputs=[chatbot, history_state]
    )

    send_btn.click(
        fn=handle_chat,
        inputs=[chat_input, history_state],
        outputs=[chat_input, chatbot, history_state]
    )

    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, history_state],
        outputs=[chat_input, chatbot, history_state]
    )

if __name__ == "__main__":
    demo.launch()