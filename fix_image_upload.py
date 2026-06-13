"""
fix_image_upload.py — Image Upload Patch (CRLF-safe, regex-based)
==================================================================
Adds image upload support to app.py.
Uses regex instead of exact string matching — immune to CRLF/emoji issues.

Run from C:\\Users\\Ezeking\\SynthForge:
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe fix_image_upload.py
"""

import re
import shutil
import sys
from pathlib import Path

APP_PATH   = Path(r"C:\Users\Ezeking\hf_space\app.py")
APP_BACKUP = Path(r"C:\Users\Ezeking\hf_space\app.py.bak_img2")

IMAGE_EXTRACT_FUNCTION = '''
def extract_image_context(uploaded_file) -> str:
    """
    Extract context from an uploaded image using Groq vision model.
    Falls back gracefully if vision is unavailable.
    """
    import base64
    groq_key = "".join(os.environ.get("GROQ_API_KEY", "").split())
    if not groq_key:
        return "[Image uploaded but GROQ_API_KEY not set]"

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
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                        {"type": "text",
                         "text": (
                             "Describe this image in detail as it relates to prompt "
                             "engineering, AI, or LLMs. If it contains code, prompts, "
                             "or system diagrams, transcribe and explain them precisely."
                         )},
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
        return f"[Image uploaded: {uploaded_file.name} — vision unavailable]"

'''


def main() -> None:
    if not APP_PATH.exists():
        print(f"ERROR: {APP_PATH} not found.")
        sys.exit(1)

    print(f"\nSynthForge Image Upload Patch (CRLF-safe)")
    print(f"Target : {APP_PATH}\n")

    # Read preserving original line endings
    raw = APP_PATH.read_bytes()
    content = raw.decode("utf-8")
    # Normalise to LF for matching, restore at end
    normalised = content.replace("\r\n", "\n")

    shutil.copy(APP_PATH, APP_BACKUP)
    print(f"Backup : {APP_BACKUP.name}\n")
    print("Applying patches:")

    # ------------------------------------------------------------------
    # Patch 1: Insert extract_image_context() before generate_answer()
    # ------------------------------------------------------------------
    if "def extract_image_context(" in normalised:
        print("  ok extract_image_context() already present — skipping")
    elif "def generate_answer(" in normalised:
        normalised = normalised.replace(
            "def generate_answer(",
            IMAGE_EXTRACT_FUNCTION + "def generate_answer(",
            1,
        )
        print("  ok Add extract_image_context() function")
    else:
        print("  x FAILED: generate_answer() anchor not found")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Patch 2: Extend file_uploader type list to include images
    # Uses regex to match regardless of exact whitespace/emoji encoding
    # ------------------------------------------------------------------
    pattern = re.compile(
        r'(type=\[")pdf(",\s*"docx",\s*"txt",\s*"md")(\])',
        re.IGNORECASE,
    )
    if 'png' in normalised and 'jpg' in normalised and 'jpeg' in normalised:
        print("  ok Image types already in file_uploader — skipping")
    elif pattern.search(normalised):
        normalised = pattern.sub(
            r'\1pdf\2, "png", "jpg", "jpeg", "webp"\3',
            normalised,
            count=1,
        )
        print("  ok Extended file_uploader to accept images")
    else:
        print("  x FAILED: file_uploader type list not found")
        print("    Search manually for: type=[\"pdf\"")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Patch 3: Route image files to extract_image_context()
    # ------------------------------------------------------------------
    IMG_ROUTE_FIND = (
        '    file_context = ""\n'
        '    had_file     = False\n'
        '    if uploaded_file:\n'
    )
    IMG_ROUTE_REPLACE = (
        '    file_context = ""\n'
        '    had_file     = False\n'
        '    if uploaded_file:\n'
        '        _img_exts = {"png", "jpg", "jpeg", "webp"}\n'
        '        _ext = uploaded_file.name.lower().rsplit(".", 1)[-1]\n'
    )

    if "_img_exts" in normalised:
        print("  ok Image routing already present — skipping")
    elif IMG_ROUTE_FIND in normalised:
        normalised = normalised.replace(IMG_ROUTE_FIND, IMG_ROUTE_REPLACE, 1)
        # Now patch the spinner/extract call to branch on image vs document
        OLD_EXTRACT = (
            '        with st.spinner(f"Processing {uploaded_file.name}..."):\n'
            '            file_context = extract_file_content(uploaded_file)'
        )
        NEW_EXTRACT = (
            '        with st.spinner(f"Processing {uploaded_file.name}..."):\n'
            '            if _ext in _img_exts:\n'
            '                file_context = extract_image_context(uploaded_file)\n'
            '            else:\n'
            '                file_context = extract_file_content(uploaded_file)'
        )
        if OLD_EXTRACT in normalised:
            normalised = normalised.replace(OLD_EXTRACT, NEW_EXTRACT, 1)
            print("  ok Image/document routing in search handler")
        else:
            # Try alternate anchor (from older UI version)
            OLD_EXTRACT_ALT = (
                '        with st.spinner(f"Extracting from {uploaded_file.name}..."):\n'
                '            file_context = extract_file_content(uploaded_file)'
            )
            NEW_EXTRACT_ALT = (
                '        with st.spinner(f"Processing {uploaded_file.name}..."):\n'
                '            if _ext in _img_exts:\n'
                '                file_context = extract_image_context(uploaded_file)\n'
                '            else:\n'
                '                file_context = extract_file_content(uploaded_file)'
            )
            if OLD_EXTRACT_ALT in normalised:
                normalised = normalised.replace(OLD_EXTRACT_ALT, NEW_EXTRACT_ALT, 1)
                print("  ok Image/document routing in search handler (alt anchor)")
            else:
                print("  ! WARNING: extract routing anchor not found — manual step needed")
    else:
        print("  x FAILED: file_context block not found")
        sys.exit(1)

    # Restore original line endings and write
    if b"\r\n" in raw:
        normalised = normalised.replace("\n", "\r\n")

    APP_PATH.write_bytes(normalised.encode("utf-8"))
    print(f"\nAll patches applied. {APP_PATH.name} updated.")
    print("\nDeploy:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "Image upload: PNG/JPG/WEBP via Groq vision"')
    print(r"  git push")


if __name__ == "__main__":
    main()
