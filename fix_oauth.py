"""
fix_oauth.py — SynthForge OAuth Resilience Patch
===================================================
Wraps st.experimental_user.is_logged_in in a try-except.
If HF OAuth is not yet active on the Space, the app falls through
gracefully and lets users access as guest instead of crashing.

Run from C:\\Users\\Ezeking\\SynthForge:
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe fix_oauth.py
"""

import shutil
import sys
from pathlib import Path

APP_PATH   = Path(r"C:\Users\Ezeking\hf_space\app.py")
APP_BACKUP = Path(r"C:\Users\Ezeking\hf_space\app.py.bak_oauth2")


def main() -> None:
    if not APP_PATH.exists():
        print(f"ERROR: {APP_PATH} not found.")
        sys.exit(1)

    raw = APP_PATH.read_bytes()
    content = raw.decode("utf-8").replace("\r\n", "\n")

    shutil.copy(APP_PATH, APP_BACKUP)
    print(f"Backup: {APP_BACKUP.name}\n")
    print("Applying patch...")

    # ------------------------------------------------------------------
    # Replace the login gate block with a try-except version
    # ------------------------------------------------------------------
    OLD_GATE = (
        "# ---------------------------------------------------------------------------\n"
        "# HF OAuth Login Gate\n"
        "# ---------------------------------------------------------------------------\n"
        "if not st.experimental_user.is_logged_in:\n"
        "    st.markdown('<div class=\"main-header\">\\U0001f525 SynthForge</div>', unsafe_allow_html=True)\n"
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
        "_pf_display   = get_user_display(_pf_username, getattr(_pf_user, 'name', None))"
    )

    NEW_GATE = (
        "# ---------------------------------------------------------------------------\n"
        "# HF OAuth Login Gate — try-except for graceful fallback\n"
        "# ---------------------------------------------------------------------------\n"
        "try:\n"
        "    _oauth_ready = hasattr(st.experimental_user, 'is_logged_in')\n"
        "    _oauth_active = _oauth_ready and st.experimental_user.is_logged_in\n"
        "except Exception:\n"
        "    _oauth_active = True   # OAuth not configured — allow guest access\n"
        "    _oauth_ready  = False\n"
        "\n"
        "if _oauth_ready and not _oauth_active:\n"
        "    st.markdown('<div class=\"main-header\">\\U0001f525 SynthForge</div>', unsafe_allow_html=True)\n"
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
        "# Resolve user identity — authenticated or guest\n"
        "try:\n"
        "    _pf_user     = st.experimental_user\n"
        "    _pf_username = getattr(_pf_user, 'preferred_username', None) or 'guest'\n"
        "    _pf_display  = get_user_display(_pf_username, getattr(_pf_user, 'name', None))\n"
        "except Exception:\n"
        "    _pf_username = 'guest'\n"
        "    _pf_display  = 'Guest'"
    )

    if OLD_GATE in content:
        content = content.replace(OLD_GATE, NEW_GATE, 1)
        print("  ok Login gate wrapped in try-except (exact match)")
    else:
        # Fallback: find the simpler unique line and do a targeted replacement
        SIMPLE_FIND = "if not st.experimental_user.is_logged_in:"
        SIMPLE_REPLACE = (
            "# try-except guards against AttributeError when OAuth not yet active\n"
            "try:\n"
            "    _oauth_ready  = hasattr(st.experimental_user, 'is_logged_in')\n"
            "    _oauth_active = _oauth_ready and st.experimental_user.is_logged_in\n"
            "except Exception:\n"
            "    _oauth_active = True\n"
            "    _oauth_ready  = False\n"
            "if _oauth_ready and not _oauth_active:"
        )
        if SIMPLE_FIND in content:
            content = content.replace(SIMPLE_FIND, SIMPLE_REPLACE, 1)
            print("  ok Login gate wrapped in try-except (simple match)")
        else:
            print("  x FAILED: login gate anchor not found in app.py")
            print("    The OAuth block may already be patched or the anchor changed.")
            sys.exit(1)

    # Also fix the user variable assignment block
    OLD_USER = (
        "# User is authenticated\n"
        "_pf_user      = st.experimental_user\n"
        "_pf_username  = _pf_user.preferred_username\n"
        "_pf_display   = get_user_display(_pf_username, getattr(_pf_user, 'name', None))"
    )
    NEW_USER = (
        "# Resolve user identity — authenticated or guest\n"
        "try:\n"
        "    _pf_user     = st.experimental_user\n"
        "    _pf_username = getattr(_pf_user, 'preferred_username', None) or 'guest'\n"
        "    _pf_display  = get_user_display(_pf_username, getattr(_pf_user, 'name', None))\n"
        "except Exception:\n"
        "    _pf_username = 'guest'\n"
        "    _pf_display  = 'Guest'"
    )
    if OLD_USER in content:
        content = content.replace(OLD_USER, NEW_USER, 1)
        print("  ok User variable block wrapped in try-except")

    # Also fix sidebar logout — only show if OAuth is actually active
    OLD_LOGOUT = (
        "    if st.button(\"Logout\", use_container_width=True):\n"
        "        st.logout()"
    )
    NEW_LOGOUT = (
        "    if _oauth_ready and _oauth_active:\n"
        "        if st.button(\"Logout\", use_container_width=True):\n"
        "            st.logout()"
    )
    if OLD_LOGOUT in content:
        content = content.replace(OLD_LOGOUT, NEW_LOGOUT, 1)
        print("  ok Logout button gated behind OAuth check")

    # Restore CRLF if original had it
    if b"\r\n" in raw:
        content = content.replace("\n", "\r\n")

    APP_PATH.write_bytes(content.encode("utf-8"))
    print(f"\nPatch applied. {APP_PATH.name} updated.")
    print("\nDeploy:")
    print(r"  cd C:\Users\Ezeking\hf_space")
    print(r"  git add app.py")
    print(r'  git commit -m "Fix: OAuth try-except — app works with or without HF login"')
    print(r"  git push")


if __name__ == "__main__":
    main()
