import re
pattern=re.compile(r'[a-z][a-z]*')
with open ("brown_nouns.txt","r",encoding="utf-8") as f:
  for line in f:
    noun=line.strip().lower()
    if re.fullmatch(pattern,noun):
      if len(noun)>=3 and noun.endswith("ies"):
        print(noun)