"""
patch_oauth.py — SynthForge HF OAuth Patch
============================================
Makes five targeted edits to hf_space/app.py:
  1. Adds user_data import
  2. Adds HF OAuth login gate (before spinner)
  3. Adds user info block to sidebar
  4. Loads persistent history on login
  5. Saves history to HF Dataset after each search

Also patches hf_space/README.md to add hf_oauth: true

Run from C:\\Users\\Ezeking\\SynthForge:
  python patch_oauth.py

Creates app.py.bak_oauth and README.md.bak before touching anything.
"""

import shutil
import sys
from pathlib import Path

APP_PATH    = Path(r"C:\Users\Ezeking\hf_space\app.py")
README_PATH = Path(r"C:\Users\Ezeking\hf_space\README.md")
APP_BACKUP    = APP_PATH.with_suffix(".py.bak_oauth")
README_BACKUP = README_PATH.with_suffix(".md.bak")


def patch_once(content: str, find: str, replace: str, label: str) -> str:
    if find not in content:
        print(f"\n  x FAILED [{label}]")
        print(f"    Anchor not found: {find[:80]!r}")
        sys.exit(1)
    print(f"  ok {label}")
    return content.replace(find, replace, 1)


def main() -> None:
    for path in [APP_PATH, README_PATH]:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    print(f"\nSynthForge HF OAuth Patch")
    print(f"App     : {APP_PATH}")
    print(f"README  : {README_PATH}\n")

    app_content    = APP_PATH.read_text(encoding="utf-8")
    readme_content = README_PATH.read_text(encoding="utf-8")

    shutil.copy(APP_PATH, APP_BACKUP)
    shutil.copy(README_PATH, README_BACKUP)
    print(f"Backups : {APP_BACKUP.name}, {README_BACKUP.name}\n")
    print("Applying patches:")

    # ------------------------------------------------------------------
    # Patch 1: Add user_data import
    # ------------------------------------------------------------------
    app_content = patch_once(
        app_content,
        "from security import detect_prompt_injection, check_rate_limit, validate_uploaded_file",
        "from security import detect_prompt_injection, check_rate_limit, validate_uploaded_file\n"
        "from user_data import load_user_history, save_user_history, get_user_display",
        "Add user_data import",
    )

    # ------------------------------------------------------------------
    # Patch 2: HF OAuth login gate + user variable — before spinner
    # ------------------------------------------------------------------
    app_content = patch_once(
        app_content,
        "# Load resources\n"
        "with st.spinner(\"Initialising SynthForge...\"):",
        # --- replacement ---
        "# ---------------------------------------------------------------------------\n"
        "# HF OAuth Login Gate\n"
        "# ---------------------------------------------------------------------------\n"
        "if not st.experimental_user.is_logged_in:\n"
        "    st.markdown('<div class=\"main-header\">🔥 SynthForge</div>', unsafe_allow_html=True)\n"
        "    st.markdown(\n"
        "        '<div class=\"sub-header\">Prompt Engineering Knowledge Synthesis Engine</div>',\n"
        "        unsafe_allow_html=True,\n"
        "    )\n"
        "    st.markdown(\"---\")\n"
        "    st.markdown(\"### Sign in to search, save history, and upload documents.\")\n"
        "    col_a, col_b, col_c = st.columns([2, 1, 2])\n"
        "    with col_b:\n"
        "        st.login()\n"
        "    st.stop()\n"
        "\n"
        "# User is authenticated\n"
        "_pf_user      = st.experimental_user\n"
        "_pf_username  = _pf_user.preferred_username\n"
        "_pf_display   = get_user_display(_pf_username, getattr(_pf_user, 'name', None))\n"
        "\n"
        "# Load resources\n"
        "with st.spinner(\"Initialising SynthForge...\"):",
        "Add HF OAuth login gate",
    )

    # ------------------------------------------------------------------
    # Patch 3: Load persistent history on first login
    # ------------------------------------------------------------------
    app_content = patch_once(
        app_content,
        "    (\"last_query\", None), (\"last_had_file\", False),\n"
        "]:\n"
        "    if key not in st.session_state:\n"
        "        st.session_state[key] = default",
        "    (\"last_query\", None), (\"last_had_file\", False),\n"
        "]:\n"
        "    if key not in st.session_state:\n"
        "        st.session_state[key] = default\n"
        "\n"
        "# Load persistent history from HF Dataset on first load for this user\n"
        "if \"_history_loaded_for\" not in st.session_state or \\\n"
        "        st.session_state[\"_history_loaded_for\"] != _pf_username:\n"
        "    st.session_state[\"search_history\"] = load_user_history(_pf_username)\n"
        "    st.session_state[\"_history_loaded_for\"] = _pf_username",
        "Load persistent history on login",
    )

    # ------------------------------------------------------------------
    # Patch 4: Add user info + logout to top of sidebar
    # ------------------------------------------------------------------
    app_content = patch_once(
        app_content,
        "with st.sidebar:\n"
        "    st.markdown(\"### ⚙️ Retrieval Settings\")",
        "with st.sidebar:\n"
        "    # User info + logout\n"
        "    st.markdown(\n"
        "        f'<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:8px;\">'\n"
        "        f'<span style=\"font-size:1.2rem;\">👤</span>'\n"
        "        f'<span style=\"font-family:Space Mono,monospace;font-size:0.82rem;'\n"
        "        f'color:#FF6B35;font-weight:700;\">{_pf_display}</span></div>',\n"
        "        unsafe_allow_html=True,\n"
        "    )\n"
        "    if st.button(\"Logout\", use_container_width=True):\n"
        "        st.logout()\n"
        "    st.markdown(\"---\")\n"
        "    st.markdown(\"### ⚙️ Retrieval Settings\")",
        "Add user info and logout to sidebar",
    )

    # ------------------------------------------------------------------
    # Patch 5: Save history to HF Dataset after each search
    # ------------------------------------------------------------------
    app_content = patch_once(
        app_content,
        "    with st.spinner(\"Generating follow-up suggestions...\"):\n"
        "        st.session_state[\"suggestions\"] = generate_suggestions(query.strip(), answer)\n"
        "\n"
        "    st.rerun()",
        "    with st.spinner(\"Generating follow-up suggestions...\"):\n"
        "        st.session_state[\"suggestions\"] = generate_suggestions(query.strip(), answer)\n"
        "\n"
        "    # Persist history to HF Dataset (background thread — non-blocking)\n"
        "    save_user_history(_pf_username, st.session_state[\"search_history\"])\n"
        "\n"
        "    st.rerun()",
        "Save history to HF Dataset after search",
    )

    APP_PATH.write_text(app_content, encoding="utf-8")
    print("  ok All app.py patches applied")

    # ------------------------------------------------------------------
    # Patch README.md — add hf_oauth: true to YAML front matter
    # ------------------------------------------------------------------
    if "hf_oauth: true" in readme_content:
        print("  ok README.md — hf_oauth already present, skipping")
    elif readme_content.startswith("---"):
        # Find the closing --- of the YAML block
        second_dash = readme_content.find("---", 3)
        if second_dash != -1:
            insert_pos = second_dash
            readme_content = (
                readme_content[:insert_pos]
                + "hf_oauth: true\n"
                + readme_content[insert_pos:]
            )
            README_PATH.write_text(readme_content, encoding="utf-8")
            print("  ok README.md — hf_oauth: true added to YAML front matter")
        else:
            print("  ! README.md — could not find YAML closing ---. Add manually: hf_oauth: true")
    else:
        print("  ! README.md — no YAML front matter found. Add hf_oauth: true manually.")

    print(f"\nAll patches applied.")
    print("\nPlace user_data.py in hf_space/, then deploy:")
    print(r"  move C:\Users\Ezeking\Downloads\user_data.py C:\Users\Ezeking\hf_space\user_data.py")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py README.md user_data.py")
    print(r'  git commit -m "HF OAuth: login gate, persistent history, user sidebar"')
    print(r"  git push")


if __name__ == "__main__":
    main()
