"""One-shot patch for src/eval/run_evals.py — three fixes."""
from pathlib import Path

src = Path("src/eval/run_evals.py")
content = src.read_text(encoding="utf-8")

# Fix 1: increase sleep from 1s to 3s
content = content.replace(
    "time.sleep(1)   # Rate limit courtesy pause",
    "time.sleep(3)   # Rate limit courtesy pause — 3s avoids Groq 429s"
)

# Fix 2: mean score over answered queries only
content = content.replace(
    '"mean_component_score": round(total_score / n, 3) if n else 0,',
    '"mean_component_score": round(total_score / answered, 3) if answered else 0,'
)

# Fix 3: only accumulate score for answered queries
content = content.replace(
    """        # Evaluate
        eval_result = check_components(answer, expected)
        score = eval_result["component_score"]
        total_score += score""",
    """        # Evaluate
        eval_result = check_components(answer, expected)
        score = eval_result["component_score"]
        if answer:  # Only count answered queries in mean
            total_score += score"""
)

# Fix 4: retry logic in generate()
old_try = '''    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content":
                     "You are SynthForge. Answer only from retrieved context. "
                     "Be concise for evaluation purposes."},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return ""'''

new_try = '''    for attempt in range(3):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content":
                         "You are SynthForge. Answer only from retrieved context. "
                         "Be concise for evaluation purposes."},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 800,
                    "temperature": 0.1,
                },
                timeout=45,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as exc:
            if resp.status_code == 429 and attempt < 2:
                wait = 15 * (2 ** attempt)
                logger.warning("Rate limited (429). Waiting %ds. Retry %d/3.", wait, attempt + 1)
                time.sleep(wait)
                continue
            logger.error("Generation failed: %s", exc)
            return ""
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return ""
    return ""'''

content = content.replace(old_try, new_try)

src.write_text(content, encoding="utf-8")
print("Done. Verify fixes:")
for needle in ["time.sleep(3)", "total_score / answered", "if answer:  # Only", "for attempt in range(3)"]:
    print(f"  {'OK' if needle in content else 'MISSING'}: {needle}")