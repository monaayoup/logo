import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Setup
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

# Use a stronger LLM and higher-quality image model
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # Truly free on HF serverless
LLM_MODEL   = "meta-llama/Llama-3.3-70B-Instruct"        # Far better prompt engineering

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

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def build_logo_prompt(conversation_history: list) -> str:
    """Use LLM with full conversation context to craft an optimized logo generation prompt."""
    temp_messages = conversation_history.copy()
    temp_messages.append({
        "role": "user",
        "content": (
            "Now write the final image generation prompt for this logo following the exact structure "
            "described in your instructions. Output ONLY the prompt, nothing else — no explanation, "
            "no preamble, no quotes around it."
        )
    })

    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=temp_messages,
        max_tokens=200,
        temperature=0.6
    )
    return resp.choices[0].message["content"].strip()


def extract_brand_name(conversation_history: list) -> str:
    """Extract the brand name from conversation."""
    temp = conversation_history.copy()
    temp.append({
        "role": "user",
        "content": "What is the brand name from our conversation? Reply with ONLY the brand name, nothing else."
    })
    resp = client.chat_completion(
        model=LLM_MODEL,
        messages=temp,
        max_tokens=20,
        temperature=0.1
    )
    return resp.choices[0].message["content"].strip()


def generate_logo(prompt: str, brand_name: str):
    """Generate the logo image and save it."""

    # SDXL-optimized prompt — no special trigger word needed
    final_prompt = (
        f"{prompt}, "
        "vector logo design, flat design, professional branding, "
        "white background, no shadows, no gradients, no noise, "
        "crisp clean lines, high resolution, sharp edges"
    )

    # FLUX doesn't support negative_prompt via HF InferenceClient directly,
    # but we front-load avoidances in the prompt text
    bad_elements = (
        "no text, no letters, no watermark, no drop shadow, no blur, "
        "no realistic photo, no 3D render, no gradient background"
    )
    final_prompt = f"{final_prompt}, {bad_elements}"

    print(f"\n🎨 Generating logo with prompt:\n→ {final_prompt}\n")

    image = client.text_to_image(
        final_prompt,
        model=IMAGE_MODEL,
    )

    safe_name = brand_name.lower().replace(" ", "_")
    filename = f"{safe_name}_logo.png"
    image.save(filename)
    image.show()
    print(f"✅ Logo saved as '{filename}'")
    return filename


# ── Main chat loop ──────────────────────────────────────────────────────────

print("\n--- 🎨 AI Logo Generator ---")
print("Describe your brand and I'll design a professional logo for you.")
print("When you're ready to generate, say 'generate', 'create', or 'go ahead'.")
print("Type 'exit' to quit.\n")

TRIGGER_WORDS = {"generate", "create", "make", "yes", "go ahead", "ready",
                 "logo now", "let's go", "do it", "build it", "make it"}

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("Bot: Good luck with your brand launch! 🚀")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        # Step 1: Normal conversational reply
        resp = client.chat_completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        bot_reply = resp.choices[0].message["content"].strip()
        messages.append({"role": "assistant", "content": bot_reply})
        print(f"\nBot: {bot_reply}\n")

        # Step 2: If user wants to generate, only proceed if LLM didn't ask for more info
        # (i.e. the bot reply doesn't contain a question asking for missing details)
        if any(word in user_input.lower() for word in TRIGGER_WORDS):

            # If the bot is still asking questions, don't generate yet
            question_indicators = ["can you please", "could you", "what is", "what are",
                                   "please provide", "i'd like to", "i need", "tell me"]
            bot_still_asking = any(phrase in bot_reply.lower() for phrase in question_indicators)

            if bot_still_asking:
                print("⚠️  Bot needs more info before generating. Please answer its questions first.\n")
            else:
                print("🏷️  Detecting brand name...")
                brand_name = extract_brand_name(messages)

                # Guard: if brand name couldn't be extracted, ask the user
                if not brand_name or brand_name.lower() in {"none", "unknown", "n/a", ""}:
                    print("Bot: I wasn't able to detect a brand name yet. What would you like to call your brand?\n")
                else:
                    print(f"   Brand: {brand_name}")
                    print("⏳ Crafting professional logo prompt...")
                    logo_prompt = build_logo_prompt(messages)
                    print(f"   Prompt built ✓")
                    generate_logo(logo_prompt, brand_name)

    except Exception as e:
        print(f"❌ Error: {e}")
        # Roll back the last user message so the loop stays clean
        if messages and messages[-1]["role"] == "user":
            messages.pop()









# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# # ── Setup ────────────────────────────────────────────────────────────────────
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# client = InferenceClient(token=HF_TOKEN)

# # Swapping to smaller, more "available" versions
# IMAGE_MODEL = "stabilityai/stable-diffusion-2-1" # Often more available than XL
# LLM_MODEL   = "meta-llama/Llama-3.1-8B-Instruct" # 8B is much lighter than 70B
# # ── Prompts ──────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT = """You are an elite brand identity designer and AI image prompt engineer with 20 years of experience.
# Your SOLE purpose is to help users generate stunning, professional logos for their brand.

# INFORMATION YOU NEED FROM THE USER
# ====================================
# 1. Brand Name       - exact name of the brand                          [REQUIRED]
# 2. Industry / Niche - e.g. tech startup, bakery, law firm              [REQUIRED]
# 3. Target Audience  - age group, lifestyle, demographics               [REQUIRED]
# 4. Brand Tone       - bold, elegant, playful, minimal, trustworthy...  [REQUIRED]
# 5. Style Preference - flat / geometric / monogram / mascot / emblem    [REQUIRED]
# 6. Color Preference - specific colors, moods, or hex codes             [REQUIRED]
# 7. Symbol / Icon    - any shape, animal, object or concept             [OPTIONAL]

# BEHAVIOR RULES
# ====================================
# - Collect the 6 required fields by asking short, friendly follow-up questions.
# - Ask ONE missing field at a time unless the user asks for all questions.
# - As soon as all 6 required fields are confirmed, DO NOT ask more questions.
#   Acknowledge what you received, tell the user you are generating their logo now, and stop.
# - If the user provides all info in one message, acknowledge it and say you are generating now.

# PROMPT ENGINEERING RULES
# ====================================
# When asked to produce the image generation prompt, use this EXACT structure:
#   "A professional vector logo for [BRAND NAME], [HIGHLY SPECIFIC ICON: shape, form, geometry],
#    [STYLE: flat design / line art / geometric / monogram], [EXACT COLOR PALETTE with hex codes],
#    [MOOD/FEEL], isolated on pure white background, no text, no letters, crisp vector edges,
#    SVG-quality, award-winning logo design, Dribbble trending, Behance featured"

# Quality rules (follow every one):
# - Icon specificity : "a geometric owl of overlapping hexagons" NOT "an owl"
# - Exact colors     : "deep obsidian #0D0D0D and electric lime #C6F135" NOT "black and green"
# - Single style     : pick exactly ONE style name
# - No raster stuff  : no gradients, drop shadows, glow, photorealism, 3D renders, bokeh, textures
# - Quality boosters : "sharp vector lines, professional branding, scalable icon, Swiss design influenced"
# - Inline avoidance : "no gradients, no shadows, no blur, no photorealism, no 3D render, no text"

# Only discuss logo and brand identity. Redirect anything else back to logo creation."""

# OPENING_MESSAGE = """
# =======================================================
#   Welcome to the AI Logo Generator!
# =======================================================
# I will design a professional logo for your brand
# automatically once I have your details.

# Please tell me about your brand:

#   Brand Name        - What is your brand called?
#   Industry / Niche  - What does your brand do?
#   Target Audience   - Who are your customers?
#   Brand Tone        - How should your brand feel?
#                       (bold / elegant / playful / minimal...)
#   Style Preference  - Any logo style you like?
#                       (flat icon / geometric / monogram / mascot...)
#   Color Preferences - Any colors or moods you have in mind?
#   Symbol / Icon     - Any specific icon or shape? (optional)

# Answer everything at once or one by one.
# I will start generating as soon as I have enough info!
# =======================================================
# """

# READINESS_CHECK_PROMPT = (
#     "Based only on the conversation so far, answer YES if ALL 6 required fields "
#     "have been clearly provided by the user: Brand Name, Industry/Niche, Target Audience, "
#     "Brand Tone, Style Preference, Color Preferences. "
#     "Answer NO if any required field is still missing or unclear. "
#     "Reply with ONLY one word: YES or NO."
# )

# # ── Conversation state ────────────────────────────────────────────────────────
# messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# # ── Helper functions ──────────────────────────────────────────────────────────
# def is_ready_to_generate(conversation_history: list) -> bool:
#     """Ask the LLM whether all 6 required brand fields have been collected."""
#     temp = conversation_history.copy()
#     temp.append({"role": "user", "content": READINESS_CHECK_PROMPT})
#     resp = client.chat_completion(
#         model=LLM_MODEL,
#         messages=temp,
#         max_tokens=5,
#         temperature=0.0  # Fully deterministic
#     )
#     answer = resp.choices[0].message["content"].strip().upper()
#     return answer.startswith("YES")


# def extract_brand_name(conversation_history: list) -> str:
#     """Extract the brand name from the conversation."""
#     temp = conversation_history.copy()
#     temp.append({
#         "role": "user",
#         "content": "What is the brand name the user provided? Reply with ONLY the brand name, nothing else."
#     })
#     resp = client.chat_completion(
#         model=LLM_MODEL,
#         messages=temp,
#         max_tokens=20,
#         temperature=0.1
#     )
#     return resp.choices[0].message["content"].strip()


# def build_logo_prompt(conversation_history: list) -> str:
#     """Build an optimized image generation prompt from the conversation context."""
#     temp = conversation_history.copy()
#     temp.append({
#         "role": "user",
#         "content": (
#             "Now write the final image generation prompt for this logo following the EXACT structure "
#             "and ALL quality rules from your instructions. Be ultra-specific about icon geometry, "
#             "name exact hex color codes, specify one design style clearly, and append all quality boosters "
#             "and avoidance phrases. Output ONLY the raw prompt with no explanation, no preamble, no quotes."
#         )
#     })
#     resp = client.chat_completion(
#         model=LLM_MODEL,
#         messages=temp,
#         max_tokens=300,
#         temperature=0.5
#     )
#     return resp.choices[0].message["content"].strip()


# def generate_logo(prompt: str, brand_name: str) -> str:
#     """Generate the logo image and save it to disk."""
#     final_prompt = (
#         f"{prompt}, "
#         "vector logo design, flat 2D design, professional branding identity, "
#         "pure white background, scalable icon mark, Swiss design influenced, "
#         "sharp crisp vector edges, high contrast, clean geometric shapes, "
#         "no text, no letters, no drop shadow, no gradient, no glow, "
#         "no photorealism, no 3D render, no blur, no watermark, no noise texture"
#     )

#     print(f"\nGenerating logo with prompt:\n  {final_prompt}\n")

#     image = client.text_to_image(final_prompt, model=IMAGE_MODEL)

#     safe_name = brand_name.lower().replace(" ", "_")
#     filename = f"{safe_name}_logo.png"
#     image.save(filename)
#     image.show()
#     print(f"Logo saved as '{filename}'")
#     return filename


# # ── Main chat loop ────────────────────────────────────────────────────────────
# print(OPENING_MESSAGE)

# while True:
#     user_input = input("You: ").strip()

#     if not user_input:
#         continue

#     if user_input.lower() in {"exit", "quit"}:
#         print("Bot: Good luck with your brand launch!")
#         break

#     messages.append({"role": "user", "content": user_input})

#     try:
#         # Step 1: Get conversational reply from the LLM
#         resp = client.chat_completion(
#             model=LLM_MODEL,
#             messages=messages,
#             max_tokens=400,
#             temperature=0.7
#         )
#         bot_reply = resp.choices[0].message["content"].strip()
#         messages.append({"role": "assistant", "content": bot_reply})
#         print(f"\nBot: {bot_reply}\n")

#         # Step 2: Auto-check if all required info is now available
#         ready = is_ready_to_generate(messages)

#         if ready:
#             print("All required info collected — starting logo generation!\n")

#             brand_name = extract_brand_name(messages)

#             if not brand_name or brand_name.lower() in {"none", "unknown", "n/a", ""}:
#                 print("Bot: I couldn't detect a brand name. What would you like to call your brand?\n")
#             else:
#                 print(f"Brand: {brand_name}")
#                 print("Crafting professional logo prompt...")
#                 logo_prompt = build_logo_prompt(messages)
#                 print("Prompt ready. Generating image...\n")
#                 generate_logo(logo_prompt, brand_name)

#     except Exception as e:
#         print(f"Error: {e}")
#         # Roll back last user message to keep conversation state clean
#         if messages and messages[-1]["role"] == "user":
#             messages.pop()