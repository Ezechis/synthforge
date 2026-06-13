"""Fix eval system prompt and max_tokens."""
from pathlib import Path

src = Path("src/eval/run_evals.py")
content = src.read_text(encoding="utf-8")

old_prompt = (
    '"You are SynthForge. Answer only from retrieved context. '
    'Be concise for evaluation purposes."'
)
new_prompt = (
    '"You are SynthForge. Answer only from retrieved context. '
    'Always cite original authors (e.g. Wei et al, Wang et al) when discussing '
    'research techniques. Include key technical terms. Aim for 150-250 words."'
)
content = content.replace(old_prompt, new_prompt)

old_tokens = '"max_tokens": 800,'
new_tokens = '"max_tokens": 1200,'
content = content.replace(old_tokens, new_tokens)

src.write_text(content, encoding="utf-8")
print("Done. Verify:")
for needle in ["Always cite original authors", '"max_tokens": 1200']:
    print(f"  {'OK' if needle in content else 'MISSING'}: {needle}")