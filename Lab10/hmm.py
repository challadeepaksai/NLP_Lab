import sys
import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple

Sentence = List[Tuple[str, str]] 
def read_tagged_corpus_wsj(path: str) -> List[Sentence]:
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            sent = []
            for tok in tokens:
                if '_' not in tok:
                    continue
                word, tag = tok.rsplit('_', 1)
                if word == '':
                    continue
                sent.append((word, tag))
            if sent:
                sentences.append(sent)
    return sentences

def k_fold_split(data: List[Sentence], K: int, seed: int = 42):
    assert K >= 2
    random.seed(seed)
    data_shuffled = data[:]
    random.shuffle(data_shuffled)
    n = len(data_shuffled)
    folds = []
    base = n // K
    extra = n % K
    start = 0
    for i in range(K):
        size = base + (1 if i < extra else 0)
        end = start + size
        test = data_shuffled[start:end]
        train = data_shuffled[:start] + data_shuffled[end:]
        folds.append((train, test))
        start = end
    return folds

class HMM:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.START = "<s>"
        self.STOP = "</s>"
        self.tag_counts = Counter()
        self.emission_counts = defaultdict(Counter)   
        self.transition_counts = defaultdict(Counter) 
        self.tags = set()
        self.vocab = set()

    def train(self, sentences: List[Sentence]):
        self.tag_counts.clear()
        self.emission_counts.clear()
        self.transition_counts.clear()
        self.tags.clear()
        self.vocab.clear()

        for sent in sentences:
            prev_tag = self.START
            self.tag_counts[prev_tag] += 1
            for word, tag in sent:
                self.tag_counts[tag] += 1
                self.emission_counts[tag][word] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.tags.add(tag)
                self.vocab.add(word)
                prev_tag = tag
            self.transition_counts[prev_tag][self.STOP] += 1
            self.tag_counts[self.STOP] += 1

        self.tags.add(self.START)
        self.tags.add(self.STOP)

    def emission_prob(self, tag: str, word: str) -> float:
        V = len(self.vocab)
        count_w_t = self.emission_counts[tag].get(word, 0)
        denom = self.tag_counts[tag] + self.alpha * (V + 1)  
        return (count_w_t + self.alpha) / denom

    def transition_prob(self, prev_tag: str, tag: str) -> float:
        N = len(self.tags)
        count_prev_to_tag = self.transition_counts[prev_tag].get(tag, 0)
        denom = sum(self.transition_counts[prev_tag].values()) + self.alpha * N
        return (count_prev_to_tag + self.alpha) / denom

def viterbi_decode(hmm: HMM, words: List[str]) -> List[str]:
    tags = [t for t in hmm.tags if t not in (hmm.START, hmm.STOP)]
    T = len(words)
    if T == 0:
        return []

    V = [dict() for _ in range(T)]

    for tag in tags:
        tp = math.log(hmm.transition_prob(hmm.START, tag))
        ep = math.log(hmm.emission_prob(tag, words[0]))
        V[0][tag] = (tp + ep, None)

    for t in range(1, T):
        for tag in tags:
            best_score = -1e300
            best_prev = None
            ep = math.log(hmm.emission_prob(tag, words[t]))
            for prev_tag in tags:
                if prev_tag not in V[t-1]:
                    continue
                prev_score = V[t-1][prev_tag][0]
                tr = math.log(hmm.transition_prob(prev_tag, tag))
                score = prev_score + tr + ep
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            if best_prev is not None:
                V[t][tag] = (best_score, best_prev)

    best_final = None
    best_score = -1e300
    for tag in tags:
        if tag not in V[T-1]:
            continue
        final_score = V[T-1][tag][0] + math.log(hmm.transition_prob(tag, hmm.STOP))
        if final_score > best_score:
            best_score = final_score
            best_final = tag

    if best_final is None:
        most_common_tag = max(hmm.tag_counts.items(), key=lambda x: x[1])[0]
        return [most_common_tag] * T

    pred = [None] * T
    pred[T-1] = best_final
    for t in range(T-1, 0, -1):
        pred[t-1] = V[t][pred[t]][1]
    return pred

def evaluate_dataset(hmm: HMM, test_sentences: List[Sentence]):
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for sent in test_sentences:
        words = [w for (w, _) in sent]
        gold = [t for (_, t) in sent]
        pred = viterbi_decode(hmm, words)
        L = min(len(gold), len(pred))
        for i in range(L):
            g = gold[i]; p = pred[i]
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1
        if len(gold) > L:
            for g in gold[L:]:
                fn[g] += 1
        if len(pred) > L:
            for p in pred[L:]:
                fp[p] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn
    }


def run_k_fold(path: str, K: int = 5, alpha: float = 1.0, seed: int = 42):
    data = read_tagged_corpus_wsj(path)
    print(f"Loaded {len(data)} sentences from {path}")
    folds = k_fold_split(data, K, seed=seed)
    fold_results = []
    agg_tp = Counter(); agg_fp = Counter(); agg_fn = Counter()

    for i, (train, test) in enumerate(folds, start=1):
        print(f"\nFold {i}/{K}: train={len(train)}, test={len(test)}")
        hmm = HMM(alpha=alpha)
        hmm.train(train)
        res = evaluate_dataset(hmm, test)
        print(f"  Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}")
        fold_results.append(res)
        agg_tp.update(res['tp']); agg_fp.update(res['fp']); agg_fn.update(res['fn'])

    total_tp = sum(agg_tp.values())
    total_fp = sum(agg_fp.values())
    total_fn = sum(agg_fn.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n=== Aggregated across folds ===")
    print(f"Aggregated Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return {
        'folds': fold_results,
        'aggregated': {'precision': precision, 'recall': recall, 'f1': f1, 'tp': agg_tp, 'fp': agg_fp, 'fn': agg_fn}
    }


if __name__ == "__main__":
    path = "wsj_pos_tagged_en.txt"
    K =3
    alpha = 1
    results = run_k_fold(path, K=K, alpha=alpha)
