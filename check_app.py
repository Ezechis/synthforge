content = open('C:/Users/Ezeking/hf_space/app.py', encoding='utf-8').read()
idx = content.find('ANTHROPIC')
print(repr(content[idx:idx+200]))