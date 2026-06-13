from pathlib import Path

src = Path("src/eval/run_evals.py")
content = src.read_text(encoding="utf-8")

old = '"Be concise for evaluation purposes."},'
new = '"Always cite original authors (e.g. Wei et al, Wang et al) when discussing research techniques. Include key technical terms. Aim for 150-250 words."},'

content = content.replace(old, new)
src.write_text(content, encoding="utf-8")

print("OK" if "Always cite original authors" in content else "MISSING")