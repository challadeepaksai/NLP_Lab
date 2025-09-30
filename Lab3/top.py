import json
from collections import Counter

def count_top_tokens(input_file, output_file, top_n=500):
    token_counter = Counter()

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            
            tokens = [tok for tok in tokens if tok != "."]

            token_counter.update(tokens)

    top_tokens = dict(token_counter.most_common(top_n))

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(top_tokens, out, ensure_ascii=False, indent=2)

    print(f"Top {top_n} tokens saved in {output_file}")


count_top_tokens("/home/deepakchalla/Desktop/NLP/Lab1/tokenized_output.txt","top_tokens.json",500)