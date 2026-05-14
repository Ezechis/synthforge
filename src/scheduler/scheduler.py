"""
PromptForge — Freshness Scheduler
===================================
Runs on your local Windows 11 machine in a dedicated CMD window.
Keeps the live Space corpus current automatically.

Schedule (WAT — Africa/Lagos):
  - Daily   03:00  arXiv ingestion  → full corpus update
  - Sunday  04:00  Reddit ingestion → full corpus update

Corpus update sequence (runs after every successful ingestion):
  1. chunk_and_embed.py       (HF_HUB_OFFLINE=1)
  2. compress_vectorstore.py
  3. build_bm25_cache.py
  4. upload_vectorstore.py
  5. HF Space restart

Usage:
  cd C:\\Users\\Ezeking\\PromptForge
  C:\\Users\\Ezeking\\AppData\\Local\\Programs\\Python\\Python311\\python.exe src/scheduler/scheduler.py

Keep the CMD window open. The process IS the scheduler — it blocks until Ctrl+C.
Logs are written to: C:\\Users\\Ezeking\\PromptForge\\logs\\scheduler.log

Installation:
  pip install apscheduler

Author: Ezechinyere Nnabugwu / DeepForge
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# ---------------------------------------------------------------------------
# Constants — edit here only
# ---------------------------------------------------------------------------

PYTHON_EXE: str = (
    r"C:\Users\Ezeking\AppData\Local\Programs\Python\Python311\python.exe"
)
PROJECT_ROOT: Path = Path(r"C:\Users\Ezeking\PromptForge")
HF_SPACE_ID: str = "ezechinnabugwu/promptforge"

# Ingestion scripts (relative to PROJECT_ROOT)
SCRIPT_INGEST_ARXIV: str = "src/ingestion/ingest_arxiv.py"
SCRIPT_INGEST_REDDIT: str = "src/ingestion/ingest_reddit.py"

# Corpus update scripts (relative to PROJECT_ROOT)
SCRIPT_CHUNK_EMBED: str = "src/processing/chunk_and_embed.py"
SCRIPT_COMPRESS: str = "deploy/compress_vectorstore.py"
SCRIPT_BUILD_BM25: str = "deploy/build_bm25_cache.py"
SCRIPT_UPLOAD: str = "deploy/upload_vectorstore.py"

# Scheduler schedule (cron format, WAT = UTC+1)
ARXIV_CRON_HOUR: int = 3    # 03:00 WAT daily
ARXIV_CRON_MINUTE: int = 0
REDDIT_CRON_DOW: str = "sun"  # Every Sunday
REDDIT_CRON_HOUR: int = 4     # 04:00 WAT
REDDIT_CRON_MINUTE: int = 0

# Grace period: if a job was missed (machine off), run it within this window
MISFIRE_GRACE_SECONDS: int = 3600  # 1 hour

# Hard timeout per script subprocess — prevents a hanging script from blocking forever
SUBPROCESS_TIMEOUT_SECONDS: int = 7200  # 2 hours

# Logging
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE: str = "scheduler.log"

# ---------------------------------------------------------------------------
# Logging setup — file + stdout
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build a subprocess environment from the current environment plus optional overrides.

    Args:
        overrides: Key-value pairs to set or override in the environment.

    Returns:
        A complete environment dict suitable for subprocess.run().
    """
    env = os.environ.copy()
    if overrides:
        env.update(overrides)
    return env


def run_script(
    script_relative: str,
    env_overrides: dict[str, str] | None = None,
) -> bool:
    """
    Execute a Python script relative to PROJECT_ROOT.

    Output streams to the terminal in real time (no capture) so you can
    see progress during long embedding runs.

    Args:
        script_relative: Script path relative to PROJECT_ROOT.
        env_overrides: Optional environment variable overrides for this call.

    Returns:
        True if the script exited with return code 0, False otherwise.
    """
    script_abs: Path = PROJECT_ROOT / script_relative
    cmd: list[str] = [PYTHON_EXE, str(script_abs)]

    logger.info("Starting script: %s", script_relative)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=_build_env(env_overrides),
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Script timed out after %ds: %s",
            SUBPROCESS_TIMEOUT_SECONDS,
            script_relative,
        )
        return False
    except FileNotFoundError as exc:
        logger.error("Script not found — check path. Script: %s | Error: %s", script_relative, exc)
        return False
    except OSError as exc:
        logger.error("OS error running script %s: %s", script_relative, exc)
        return False

    if result.returncode != 0:
        logger.error(
            "Script exited with code %d: %s", result.returncode, script_relative
        )
        return False

    logger.info("Script completed: %s", script_relative)
    return True


def restart_space() -> None:
    """
    Trigger a restart of the live Hugging Face Space.

    Uses the HF_TOKEN already present in the environment (set as Windows
    environment variable or in HF Secrets). If the token is missing, the
    restart is skipped with an error log — the upload will still be live
    on the next cold start or UptimeRobot ping.
    """
    hf_token: str | None = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error(
            "HF_TOKEN not found in environment — Space restart skipped. "
            "Set it with: setx HF_TOKEN your_token_here"
        )
        return

    try:
        from huggingface_hub import HfApi  # type: ignore[import]

        api = HfApi(token=hf_token)
        api.restart_space(repo_id=HF_SPACE_ID)
        logger.info("HF Space restart triggered: %s", HF_SPACE_ID)
    except ImportError:
        logger.error("huggingface_hub not installed — cannot restart Space.")
    except Exception as exc:
        logger.error("Space restart failed: %s", exc)


def corpus_update_sequence() -> bool:
    """
    Run the four-step corpus update pipeline after ingestion.

    Order mirrors the manual SOP exactly:
      Step 1: chunk_and_embed.py   (HF_HUB_OFFLINE=1 — avoids Windows DNS issue)
      Step 2: compress_vectorstore.py
      Step 3: build_bm25_cache.py
      Step 4: upload_vectorstore.py
      Step 5: HF Space restart

    Aborts at the first failure — a partial upload is worse than no upload.

    Returns:
        True if all steps succeeded, False if any step failed.
    """
    logger.info("=== Corpus update sequence starting ===")

    steps: list[tuple[str, dict[str, str] | None]] = [
        (SCRIPT_CHUNK_EMBED, {"HF_HUB_OFFLINE": "1"}),
        (SCRIPT_COMPRESS, None),
        (SCRIPT_BUILD_BM25, None),
        (SCRIPT_UPLOAD, {"HF_HUB_OFFLINE": "0"}),
    ]

    for script, env_overrides in steps:
        success = run_script(script, env_overrides=env_overrides)
        if not success:
            logger.error("Corpus update aborted at step: %s", script)
            return False

    restart_space()
    logger.info("=== Corpus update sequence complete ===")
    return True


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------


def job_arxiv_daily() -> None:
    """
    Daily job: fetch new arXiv papers on prompt engineering topics,
    then run the full corpus update sequence.

    Runs at ARXIV_CRON_HOUR:ARXIV_CRON_MINUTE WAT (default 03:00).
    """
    start = datetime.now()
    logger.info("=== arXiv daily job started: %s ===", start.isoformat())

    ingestion_ok = run_script(SCRIPT_INGEST_ARXIV)

    if not ingestion_ok:
        logger.error(
            "arXiv ingestion failed — corpus update skipped. "
            "Check the ingestion script logs for details."
        )
        return

    update_ok = corpus_update_sequence()

    elapsed = (datetime.now() - start).total_seconds() / 60
    if update_ok:
        logger.info("arXiv daily job completed in %.1f minutes.", elapsed)
    else:
        logger.error("arXiv daily job: corpus update failed. Elapsed: %.1f min.", elapsed)


def job_reddit_weekly() -> None:
    """
    Weekly job: re-crawl the 5 target subreddits through all 4 quality gates,
    then run the full corpus update sequence.

    Runs at REDDIT_CRON_HOUR:REDDIT_CRON_MINUTE WAT on REDDIT_CRON_DOW
    (default Sunday 04:00).
    """
    start = datetime.now()
    logger.info("=== Reddit weekly job started: %s ===", start.isoformat())

    ingestion_ok = run_script(SCRIPT_INGEST_REDDIT)

    if not ingestion_ok:
        logger.error(
            "Reddit ingestion failed — corpus update skipped. "
            "Check the ingestion script logs for details."
        )
        return

    update_ok = corpus_update_sequence()

    elapsed = (datetime.now() - start).total_seconds() / 60
    if update_ok:
        logger.info("Reddit weekly job completed in %.1f minutes.", elapsed)
    else:
        logger.error("Reddit weekly job: corpus update failed. Elapsed: %.1f min.", elapsed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure and start the blocking APScheduler instance."""

    scheduler = BlockingScheduler(timezone="Africa/Lagos")

    # --- Daily arXiv ---
    scheduler.add_job(
        job_arxiv_daily,
        trigger=CronTrigger(
            hour=ARXIV_CRON_HOUR,
            minute=ARXIV_CRON_MINUTE,
            timezone="Africa/Lagos",
        ),
        id="arxiv_daily",
        name="Daily arXiv ingestion",
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
        coalesce=True,  # If multiple misfires, run once instead of many times
    )

    # --- Weekly Reddit ---
    scheduler.add_job(
        job_reddit_weekly,
        trigger=CronTrigger(
            day_of_week=REDDIT_CRON_DOW,
            hour=REDDIT_CRON_HOUR,
            minute=REDDIT_CRON_MINUTE,
            timezone="Africa/Lagos",
        ),
        id="reddit_weekly",
        name="Weekly Reddit ingestion",
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
        coalesce=True,
    )

    logger.info("=" * 60)
    logger.info("PromptForge Freshness Scheduler — STARTED")
    logger.info("  arXiv:  daily at %02d:%02d WAT", ARXIV_CRON_HOUR, ARXIV_CRON_MINUTE)
    logger.info(
        "  Reddit: every %s at %02d:%02d WAT",
        REDDIT_CRON_DOW.capitalize(),
        REDDIT_CRON_HOUR,
        REDDIT_CRON_MINUTE,
    )
    logger.info("  Log:    %s", LOG_DIR / LOG_FILE)
    logger.info("  Press Ctrl+C to stop cleanly.")
    logger.info("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (Ctrl+C).")
        scheduler.shutdown(wait=False)
    except Exception as exc:
        logger.error("Scheduler crashed unexpectedly: %s", exc)
        scheduler.shutdown(wait=False)
        raise


if __name__ == "__main__":
    main()
