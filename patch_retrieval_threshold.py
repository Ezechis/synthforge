# Filters out low-scoring retrieved chunks before they reach the LLM.
# Scores below -6.5 are false positives (React Native matching ReAct, etc.)
# Fallback ensures minimum 3 chunks always reach the LLM.

import sys
from pathlib import Path

APP = Path(r"C:\Users\Ezeking\hf_space\app.py")

old = (
    '        # Generate\n'
    '        answer = generate_answer(query.strip(), results)'
)
new = (
    '        # Filter low-confidence retrievals before generation\n'
    '        MIN_SCORE = -6.5\n'
    '        filtered = [r for r in results if r["score"] > MIN_SCORE]\n'
    '        results = filtered if len(filtered) >= 3 else results[:3]\n'
    '\n'
    '        # Generate\n'
    '        answer = generate_answer(query.strip(), results)'
)

content = APP.read_text(encoding="utf-8")
if content.count(old) != 1:
    print(f"ERROR: {content.count(old)} matches"); sys.exit(1)
APP.write_text(content.replace(old, new, 1), encoding="utf-8")
print("OK — retrieval threshold -6.5 applied")