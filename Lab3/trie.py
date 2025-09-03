import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def split_word(self, word):
        node = self.root
        max_branches = 0
        split_pos = 0
        
        for i, char in enumerate(word):
            if char in node.children:
                node = node.children[char]
                branches = len(node.children)
                if branches > max_branches:
                    max_branches = branches
                    split_pos = i + 1
            else:
                break
        
        if split_pos == 0:
            return f"{word}={word}+"
        
        stem = word[:split_pos]
        suffix = word[split_pos:]
        return f"{word}={stem}+{suffix}"

# Usage
pretrie = PrefixTrie()
suftrie = PrefixTrie()
nouns = set()
pat = re.compile(r'[a-z][a-z]*')

with open("/home/deepakchalla/Desktop/NLP/Lab2/brown_nouns.txt", "r", encoding="utf-8") as f:
    for line in f:
        noun = line.strip().lower()
        if noun and re.fullmatch(pat, noun):
            nouns.add(noun)

for noun in nouns:
    pretrie.insert(noun)
    suftrie.insert(noun[::-1]) 

with open("suffix_trie.txt", "w", encoding="utf-8") as f:
    for noun in nouns:
        reversed_noun = noun[::-1]
        suf_result = suftrie.split_word(reversed_noun)
        parts = suf_result.split('=')[1].split('+')
        reversed_stem = parts[0][::-1]
        reversed_suffix = parts[1][::-1] if parts[1] else ""
        final_result = f"{noun}={reversed_suffix}+{reversed_stem}"
        f.write(f"{final_result}\n")

with open("prefix_trie.txt", "w", encoding="utf-8") as f:
    for noun in nouns:
        pre_result = pretrie.split_word(noun)
        f.write(f"{pre_result}\n")