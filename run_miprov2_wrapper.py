"""
MIPROv2 wrapper — forces GROQ_API_KEY into environment before import.
Run this instead of run_miprov2.py when set/setx fails.

Usage:
    python run_miprov2_wrapper.py
"""
import os
import sys

# Force the key into the environment BEFORE any other import
# This bypasses all CMD set/setx issues
os.environ["GROQ_API_KEY"] = ""

# Verify it's set
key = os.environ.get("GROQ_API_KEY", "")
if not key or len(key) < 20:
    print("ERROR: GROQ_API_KEY not set correctly.")
    sys.exit(1)

print(f"GROQ_API_KEY confirmed: {key[:8]}...{key[-4:]} ({len(key)} chars)")

# Now add the src directory to path and run MIPROv2
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Execute run_miprov2.py in the same process with key already in env
mipro_path = os.path.join(script_dir, "src", "optimization", "run_miprov2.py")
print(f"Running: {mipro_path}")

with open(mipro_path, "r", encoding="utf-8") as f:
    source = f.read()

exec(compile(source, mipro_path, "exec"), {"__name__": "__main__", "__file__": mipro_path})
