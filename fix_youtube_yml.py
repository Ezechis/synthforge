from pathlib import Path

yml = Path(".github/workflows/youtube_batch.yml")
content = yml.read_text(encoding="utf-8")

download_step = '''
      - name: Download YouTube progress file
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
        run: |
          python -c "
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
token = os.environ['HF_TOKEN']
repo = os.environ.get('HF_DATASET_REPO', 'ezechinnabugwu/synthforge-vectorstore')
dest = Path('data/youtube_progress.json')
dest.parent.mkdir(parents=True, exist_ok=True)
try:
    hf_hub_download(repo_id=repo, repo_type='dataset', filename='youtube_progress.json', token=token, local_dir=str(dest.parent))
    print('Progress file downloaded.')
except EntryNotFoundError:
    print('No progress file yet — starting fresh.')
except Exception as exc:
    print(f'Warning: {exc}')
"

'''

upload_step = '''
      - name: Upload YouTube progress file
        if: always()
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
        run: |
          python -c "
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
token = os.environ['HF_TOKEN']
repo = os.environ.get('HF_DATASET_REPO', 'ezechinnabugwu/synthforge-vectorstore')
f = Path('data/youtube_progress.json')
if not f.exists():
    print('No progress file to upload.')
    sys.exit(0)
HfApi(token=token).upload_file(path_or_fileobj=str(f), path_in_repo='youtube_progress.json', repo_id=repo, repo_type='dataset', commit_message='Update YouTube transcription progress')
print('Progress file uploaded.')
"

'''

anchor_download = "      - name: Run YouTube transcription batch (Groq Whisper)"
anchor_upload   = "      - name: Embed new transcript chunks"

content = content.replace(anchor_download, download_step + anchor_download)
content = content.replace(anchor_upload,   upload_step   + anchor_upload)

yml.write_text(content, encoding="utf-8")

ok1 = "Download YouTube progress file" in content
ok2 = "Upload YouTube progress file"   in content
ok3 = "if: always()"                   in content
print(f"Download step: {'OK' if ok1 else 'MISSING'}")
print(f"Upload step:   {'OK' if ok2 else 'MISSING'}")
print(f"if: always():  {'OK' if ok3 else 'MISSING'}")