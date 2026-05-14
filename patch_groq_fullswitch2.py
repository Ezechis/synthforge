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


patch(
    'ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"\n'
    'ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"',
    'GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"\n'
    'GROQ_MODEL: str = "llama-3.3-70b-versatile"',
    "Fix 1 API constants"
)

patch(
    '    anthropic_key = "".join(os.environ.get("ANTHROPIC_API_KEY", "").split())\n'
    '    if not anthropic_key:\n'
    '        return "ANTHROPIC_API_KEY not set in Space secrets."',
    '    groq_key = "".join(os.environ.get("GROQ_API_KEY", "").split())\n'
    '    if not groq_key:\n'
    '        return "GROQ_API_KEY not set in Space secrets."',
    "Fix 2 key variable"
)

patch(
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
    '                "max_tokens": 1500,\n'
    '                "temperature": 0.1,\n'
    '            },\n'
    '            timeout=60,\n'
    '        )\n'
    '        response.raise_for_status()\n'
    '        return response.json()["choices"][0]["message"]["content"]',
    "Fix 3 API call"
)

patch(
    'm4.metric("Model", "Claude Sonnet")',
    'm4.metric("Model", "Llama 3.3 70B")',
    "Fix 4 model label"
)

print("\nGroq llama-3.3-70b fully wired.")