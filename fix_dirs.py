import pathlib
for f in ['.github/workflows/daily_arxiv.yml','.github/workflows/weekly_reddit.yml','.github/workflows/youtube_batch.yml']:
    c=open(f,encoding='utf-8').read()
    old='      - name: Install Python dependencies\n        run: pip install -r requirements_actions.txt'
    new=old+'\n\n      - name: Create required directories\n        run: mkdir -p logs data/raw data/transcripts data/audio_tmp'
    if 'Create required directories' in c:
        print('Already fixed:',f)
    elif old in c:
        open(f,'w',encoding='utf-8').write(c.replace(old,new,1))
        print('Fixed:',f)
    else:
        print('Anchor not found:',f)
