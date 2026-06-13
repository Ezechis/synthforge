"""
patch_image_upload.py — SynthForge Image Upload Patch
=======================================================
Adds image upload support to the file uploader in app.py.
Images are converted to base64 and included in the query context.
The generation layer passes them to the LLM (Claude Sonnet 4.6 natively
handles vision; Groq/Llama receives a text description instead).

Changes made to app.py:
  1. Extends file_uploader to accept image types (PNG, JPG, WEBP)
  2. Adds extract_image_context() — converts image to base64 or
     generates a text description via Groq vision-capable model
  3. Injects image context into generate_answer() alongside file_context

Run from C:\\Users\\Ezeking\\SynthForge:
  python patch_image_upload.py

Run AFTER patch_oauth.py has been applied.
Creates app.py.bak_image before touching anything.
"""

import shutil
import sys
from pathlib import Path

APP_PATH   = Path(r"C:\Users\Ezeking\hf_space\app.py")
APP_BACKUP = Path(r"C:\Users\Ezeking\hf_space\app.py.bak_image")


def patch_once(content: str, find: str, replace: str, label: str) -> str:
    if find not in content:
        print(f"\n  x FAILED [{label}]")
        print(f"    Anchor not found: {find[:80]!r}")
        sys.exit(1)
    print(f"  ok {label}")
    return content.replace(find, replace, 1)


IMAGE_EXTRACT_FUNCTION = '''
def extract_image_context(uploaded_file) -> str:
    """
    Extract context from an uploaded image for use in prompt generation.

    For Groq (current model): sends image to llava or describes it via
    a vision-capable endpoint. Falls back to a placeholder if unavailable.
    For Claude Sonnet 4.6 (future): passes base64 directly to the API.

    Args:
        uploaded_file: Streamlit UploadedFile with image content.

    Returns:
        Text description of the image suitable for prompt context.
    """
    import base64
    groq_key = "".join(os.environ.get("GROQ_API_KEY", "").split())
    if not groq_key:
        return "[Image uploaded but no API key available for vision processing]"

    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    mime_map = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    b64_image = base64.b64encode(raw_bytes).decode("utf-8")

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}",
                     "Content-Type": "application/json"},
            json={
                "model": "llama-3.2-11b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail as it relates to "
                                "prompt engineering, AI systems, or LLM usage. "
                                "If it contains code, prompts, or system diagrams, "
                                "transcribe and explain them precisely."
                            ),
                        },
                    ],
                }],
                "max_tokens": 600,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
        return f"[IMAGE CONTEXT]\\n{description}"
    except Exception as exc:
        logger.warning("Image vision processing failed: %s", exc)
        return f"[Image uploaded: {uploaded_file.name} — vision processing unavailable]"

'''


def main() -> None:
    if not APP_PATH.exists():
        print(f"ERROR: {APP_PATH} not found.")
        sys.exit(1)

    print(f"\nSynthForge Image Upload Patch")
    print(f"Target  : {APP_PATH}\n")

    content = APP_PATH.read_text(encoding="utf-8")
    shutil.copy(APP_PATH, APP_BACKUP)
    print(f"Backup  : {APP_BACKUP.name}\n")
    print("Applying patches:")

    # ------------------------------------------------------------------
    # Patch 1: Insert extract_image_context() before generate_answer()
    # ------------------------------------------------------------------
    content = patch_once(
        content,
        "def generate_answer(",
        IMAGE_EXTRACT_FUNCTION + "def generate_answer(",
        "Add extract_image_context() function",
    )

    # ------------------------------------------------------------------
    # Patch 2: Extend file_uploader to accept image types
    # ------------------------------------------------------------------
    content = patch_once(
        content,
        "\"📎 Attach a document:\",\n"
        "        type=[\"pdf\", \"docx\", \"txt\", \"md\"],",
        "\"📎 Attach a document or image:\",\n"
        "        type=[\"pdf\", \"docx\", \"txt\", \"md\", \"png\", \"jpg\", \"jpeg\", \"webp\"],",
        "Extend file uploader to accept images",
    )

    # ------------------------------------------------------------------
    # Patch 3: Route image files to extract_image_context()
    # ------------------------------------------------------------------
    content = patch_once(
        content,
        "    file_context = \"\"\n"
        "    had_file     = False\n"
        "    if uploaded_file:\n"
        "        with st.spinner(f\"Extracting from {uploaded_file.name}...\"):\n"
        "            file_context = extract_file_content(uploaded_file)",
        "    file_context = \"\"\n"
        "    had_file     = False\n"
        "    if uploaded_file:\n"
        "        _img_exts = {\"png\", \"jpg\", \"jpeg\", \"webp\"}\n"
        "        _ext = uploaded_file.name.lower().rsplit(\".\", 1)[-1]\n"
        "        with st.spinner(f\"Processing {uploaded_file.name}...\"):\n"
        "            if _ext in _img_exts:\n"
        "                file_context = extract_image_context(uploaded_file)\n"
        "            else:\n"
        "                file_context = extract_file_content(uploaded_file)",
        "Route image files to extract_image_context()",
    )

    APP_PATH.write_text(content, encoding="utf-8")
    print(f"\nAll patches applied. {APP_PATH.name} updated.")
    print("\nDeploy:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "Image upload: PNG/JPG/WEBP support via Groq vision"')
    print(r"  git push")


if __name__ == "__main__":
    main()
