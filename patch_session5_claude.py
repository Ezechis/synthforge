# Session 5 — Switch generation model to Claude Sonnet 4.6
# Replaces Groq API with Anthropic API in hf_space/app.py
# Lifts word cap 150->400, chunks 6->8 to exploit Claude's context window

import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")


def patch(old, new, label):
    content = APP.read_text(encoding="utf-8")
    count = content.count(old)
    if count == 0:
        print(f"ERROR [{label}]: string not found"); sys.exit(1)
    if count > 1:
        print(f"ERROR [{label}]: found {count} times"); sys.exit(1)
    APP.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"OK    [{label}]")


# ── Patch 1: Replace API constants ───────────────────────────────────────────
patch(
    'GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"\n'
    'GROQ_MODEL: str = "llama-3.1-8b-instant"',
    'ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"\n'
    'ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"',
    "Patch 1 API constants"
)

# ── Patch 2: Lift word cap 150->400 and chunks 6->8 ──────────────────────────
patch(
    '    for chunk in retrieved_chunks[:6]:\n'
    '        meta = chunk["metadata"]\n'
    '        source_label = (\n'
    '            f"[{meta.get(\'source\',\'unknown\').upper()} | "\n'
    '            f"Author: {meta.get(\'author\',\'\') or \'N/A\'} | "\n'
    '            f"{meta.get(\'title\',\'\')[:60]} | "\n'
    '            f"credibility: {meta.get(\'credibility_tier\',\'unknown\')}]"\n'
    '        )\n'
    '        block = f"{source_label}\\n{chr(32).join(chunk[\'text\'].split()[:300])}"\n'
    '        block_words = len(block.split())\n'
    '        if word_count + block_words > MAX_CONTEXT_WORDS:\n'
    '            break\n'
    '        block=" ".join(block.split()[:150]);context_parts.append(block)\n'
    '        word_count += block_words',
    '    for chunk in retrieved_chunks[:8]:\n'
    '        meta = chunk["metadata"]\n'
    '        source_label = (\n'
    '            f"[{meta.get(\'source\',\'unknown\').upper()} | "\n'
    '            f"Author: {meta.get(\'author\',\'\') or \'N/A\'} | "\n'
    '            f"{meta.get(\'title\',\'\')[:60]} | "\n'
    '            f"credibility: {meta.get(\'credibility_tier\',\'unknown\')}]"\n'
    '        )\n'
    '        block = f"{source_label}\\n{chr(32).join(chunk[\'text\'].split()[:300])}"\n'
    '        block_words = len(block.split())\n'
    '        if word_count + block_words > MAX_CONTEXT_WORDS:\n'
    '            break\n'
    '        block=" ".join(block.split()[:400]);context_parts.append(block)\n'
    '        word_count += block_words',
    "Patch 2 word cap and chunk count"
)

# ── Patch 3: Replace generate_answer function body ────────────────────────────
patch(
    '    groq_key = os.environ.get("GROQ_API_KEY", "").strip()\n'
    '    if not groq_key:\n'
    '        return "GROQ_API_KEY not set in Space secrets."',
    '    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()\n'
    '    if not anthropic_key:\n'
    '        return "ANTHROPIC_API_KEY not set in Space secrets."',
    "Patch 3 API key variable"
)

# ── Patch 4: Replace the API call ────────────────────────────────────────────
patch(
    '        response = requests.post(\n'
    '            GROQ_API_URL,\n'
    '            headers={\n'
    '                "Authorization": f"Bearer {groq_key}",\n'
    '                "Content-Type": "application/json",\n'
    '            },\n'
    '            json={\n'
    '                "model": GROQ_MODEL,\n'
    '                "messages": [\n'
    '                    {"role": "system", "content": SYSTEM_PROMPT},\n'
    '                    {"role": "user", "content": user_message},\n'
    '                ],\n'
    '                "max_tokens": 800,\n'
    '                "temperature": 0.1,\n'
    '            },\n'
    '            timeout=60,\n'
    '        )\n'
    '        response.raise_for_status()\n'
    '        return response.json()["choices"][0]["message"]["content"]',
    '        response = requests.post(\n'
    '            ANTHROPIC_API_URL,\n'
    '            headers={\n'
    '                "x-api-key": anthropic_key,\n'
    '                "anthropic-version": "2023-06-01",\n'
    '                "content-type": "application/json",\n'
    '            },\n'
    '            json={\n'
    '                "model": ANTHROPIC_MODEL,\n'
    '                "max_tokens": 1500,\n'
    '                "temperature": 0.1,\n'
    '                "system": SYSTEM_PROMPT,\n'
    '                "messages": [\n'
    '                    {"role": "user", "content": user_message},\n'
    '                ],\n'
    '            },\n'
    '            timeout=60,\n'
    '        )\n'
    '        response.raise_for_status()\n'
    '        return response.json()["content"][0]["text"]',
    "Patch 4 API call"
)

# ── Patch 5: Fix model label in UI metric ─────────────────────────────────────
patch(
    'm4.metric("Model", GROQ_MODEL.split("-")[0].upper())',
    'm4.metric("Model", "Claude Sonnet")',
    "Patch 5 model label"
)

print("\nSession 5 complete — Claude Sonnet 4.6 wired in.")