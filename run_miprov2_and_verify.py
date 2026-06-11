"""
Wrapper script to run MIPROv2 optimization followed by pipeline verification.
Executes run_miprov2_final.py then verify_all_pipelines.py in sequence.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_and_install_dependencies():
    """Check and install missing dependencies from requirements.txt."""
    print("\n" + "="*60)
    print("Checking and installing dependencies...")
    print("="*60)
    
    # Map package names in requirements.txt to their import names
    import_map = {
        "dspy-ai": "dspy",
        "huggingface-hub": "huggingface_hub",
        "python-dotenv": "dotenv",
        "rank-bm25": "rank_bm25",
        "sentence-transformers": "sentence_transformers",
        "youtube-transcript-api": "youtube_transcript_api",
    }
    
    missing_packages = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Extract package name (remove version specifiers)
                package = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                if not package:
                    continue
                
                # Determine import name
                import_name = import_map.get(package, package)
                
                # Check if package is installed
                try:
                    __import__(import_name)
                except ImportError:
                    missing_packages.append(package)
    except FileNotFoundError:
        print("[WARN] requirements.txt not found, skipping dependency check")
        return
    
    if missing_packages:
        print(f"[INFO] Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  [OK] Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] Failed to install {package}: {e}")
                print(f"       Please install manually: pip install {package}")
    else:
        print("[OK] All dependencies are satisfied")

def run_script(script_path, description):
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for MIPROv2
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode != 0:
            print(f"\n{description} FAILED with exit code {result.returncode}")
            return False
        else:
            print(f"\n{description} COMPLETED SUCCESSFULLY")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"\n{description} TIMEOUT (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"\n{description} ERROR: {e}")
        return False

def main():
    """Run MIPROv2 optimization then verification."""
    print("SynthForge MIPROv2 + Verification Pipeline")
    print("==========================================")
    
    # Check we're in the right directory
    if not Path("src").exists() or not Path("data").exists():
        print("ERROR: Please run this script from the PromptForge root directory")
        print("       (where src/ and data/ directories are present)")
        sys.exit(1)
    
    # Step 0: Check and install dependencies
    check_and_install_dependencies()
    
    # Step 1: Run MIPROv2 optimization
    success1 = run_script(
        "run_miprov2_final.py",
        "MIPROv2 Optimization"
    )
    
    if not success1:
        print("\n" + "!"*60)
        print("MIPROv2 OPTIMIZATION FAILED")
        print("Please fix the issues above before proceeding")
        print("!"*60)
        sys.exit(1)
    
    # Step 2: Run pipeline verification
    success2 = run_script(
        "verify_all_pipelines.py",
        "Pipeline Verification"
    )
    
    if not success2:
        print("\n" + "!"*60)
        print("PIPELINE VERIFICATION FAILED")
        print("Please fix the issues above")
        print("!"*60)
        sys.exit(1)
    
    # Both succeeded
    print("\n" + "="*60)
    print("🎉 MIPROv2 EVALS COMPLETED SUCCESSFULLY 🎉")
    print("="*60)
    print("The optimized prompt has been generated and verified.")
    print("Check data/optimization/optimized_prompt_latest.txt for results.")
    print("="*60)

if __name__ == "__main__":
    main()
