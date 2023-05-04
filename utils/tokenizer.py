from collections import defaultdict
import json
from tqdm import tqdm

def get_scores(splits: dict, corpus: list[str])->dict:
    subword_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    for word in corpus:
        n = len(splits[word])
        for i, c in enumerate(splits[word]):
            subword_freq[c] += 1
            if i + 1 < n:
                pair_freq[(c, splits[word][i + 1])] += 1
    # scores = { (p0, p1): p_f / (subword_freq[p0] * subword_freq[p1]) for (p0, p1), p_f in pair_freq.items() }
    # scores = { (p0, p1): p_f / (subword_freq[p0]) for (p0, p1), p_f in pair_freq.items() }
    scores = { (p0, p1): p_f for (p0, p1), p_f in pair_freq.items() }
    return scores

class WordpieceTokenizer:
    def __init__(self) -> None:
        self.vocab = []
        self.unk_token = '<unk>'

    def load(self, path: str):
        with open(path, 'r') as f:
            self.vocab = json.load(f)

    def train(self, corpus: list[str], vocab_size: int, unk_token: str = '<unk>'):
        """
        corpus: a list of words from pre-tokenization
        """
        self.vocab = [unk_token]
        vocab_set = {unk_token}
        splits = {}
        words = set()
        for word in corpus:
            split = []
            for id, c in enumerate(word):
                if id != 0:
                    c = '##' + c
                if c not in vocab_set:
                    self.vocab.append(c)
                    vocab_set.add(c)
                split.append(c)
            splits[word] = split
            words.add(word)

        # print(splits)

        while len(self.vocab) < vocab_size:
            
            if len(self.vocab) % 100 == 0:
                print(best_p, best_s)
                print(len(self.vocab))

            scores = get_scores(splits, corpus)
            if len(scores) == 0:
                break

            best_p, best_s = '', None
            for p, s in scores.items():
                if best_s is None or s > best_s:
                    best_p = p
                    best_s = s
            if best_s == 1:
                break
            new_subword = best_p[0] + best_p[1][2:]
            self.vocab.append(new_subword)
            
            for word in words:
                split = splits[word]
                new_split = []
                ignore = -1
                for i in range(len(split)):
                    if i == ignore:
                        continue
                    sw = split[i]
                    if i + 1 < len(split) and (sw + split[i + 1][2:] == new_subword):
                        new_split.append(new_subword)
                        ignore = i + 1
                    else:
                        new_split.append(sw)
                splits[word] = new_split       

            # break             

    def tokenize(self, word: str)->list[str]:
        """
        word: a word produced from pre-tokenization
        """
        toks = []
        first = True
        while len(word):
            tok = None
            pos = None
            for i in range(len(word)):
                prefix = '##' + word[:i + 1] if not first else word[:i + 1]
                if prefix in self.vocab:
                    tok = prefix
                    pos = i + 1
            if tok is None:
                return [self.unk_token]
            word = word[pos:]
            toks.append(tok)
            first = False
        return toks
