content=open(r'C:/Users/Ezeking/hf_space/app.py',encoding='utf-8').read()
s=content.find('SYSTEM_PROMPT')
e=content.find('"""',s+20)+3
open('prompt_out.txt','w',encoding='utf-8').write(content[s:e])
print('Done')
