"""A byte-level BPE tokenizer from scratch: the same idea as GPT's tokenizers, in plain Python, no installs.

    python3 bpe.py                  # train on Alice's Adventures in Wonderland, watch the merges
    python3 bpe.py 2000 "Hello"     # more merges, then tokenize your own text

Text -> UTF-8 bytes (IDs 0-255) -> merge the most common neighbouring pair into a new ID -> repeat.
"""
import re, sys, time
from collections import Counter

# Cut text into chunks first (a word with its leading space, a number, punctuation), as GPT-2 does,
# so a merge never glues the end of one word to the start of the next.
SPLIT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


def count_pairs(ids, counts, times=1):
    for pair in zip(ids, ids[1:]):            # every neighbouring pair, e.g. (115, 116) for "s","t"
        counts[pair] += times


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)                # the pair becomes one new token
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self):
        self.merges = {}                                    # (a, b) -> new ID, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}    # ID -> the bytes it stands for
        self.special = {}                                   # "<|end|>" -> ID: control tokens, never made by merging

    def train(self, text, vocab_size, log=None):
        chunks = Counter(SPLIT.findall(text))               # each distinct chunk once, with how often it appears
        words = {c: list(c.encode("utf-8")) for c in chunks}
        for new_id in range(256, vocab_size):
            counts = Counter()
            for c, ids in words.items():
                count_pairs(ids, counts, chunks[c])
            if not counts:
                break                                       # nothing left to merge
            best = max(counts, key=counts.get)              # the most common neighbouring pair
            for c in words:
                words[c] = merge(words[c], best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if log:
                log(new_id, best, counts[best], self.vocab[new_id])
        return self

    def add_special(self, *names):
        for name in names:
            self.special[name] = len(self.vocab) + len(self.special)

    def encode(self, text, allow_special=False):
        if self.special and allow_special:                  # cut special tokens out first, they map straight to IDs
            parts = re.split("(" + "|".join(map(re.escape, self.special)) + ")", text)
            return [i for p in parts for i in ([self.special[p]] if p in self.special else self.encode(p))]
        out = []
        for chunk in SPLIT.findall(text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) > 1:                             # replay the merges, earliest-learned first
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            out += ids
        return out

    def decode(self, ids):
        names = {i: s.encode("utf-8") for s, i in self.special.items()}
        return b"".join(names.get(i) or self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def pieces(self, ids):
        names = {i: s for s, i in self.special.items()}
        return [names.get(i) or self.vocab[i].decode("utf-8", errors="replace") for i in ids]


def show(b):                                                 # a token's bytes, with its leading space made visible
    return repr(b.decode("utf-8", errors="replace").replace(" ", "␣"))


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    book = open("data/alice.txt", encoding="utf-8").read()
    cut = book.index("CHAPTER XII.")
    train, test = book[:cut], book[cut:]                    # chapters I-XI to learn from, chapter XII kept unseen
    print(f"Alice's Adventures in Wonderland, chapters I-XI: {len(train):,} characters, {len(train.encode()):,} bytes\n")
    log = lambda i, pair, n, b: (i < 266 or i % 100 == 0) and print(f"merge {i - 255:4}: {pair} -> {i:4} {show(b):14} seen {n:,} times")
    t0 = time.time()
    tok = BPE().train(train, vocab_size, log)
    print(f"\n{len(tok.merges)} merges in {time.time() - t0:.1f} s")

    sample = sys.argv[2] if len(sys.argv) > 2 else "How many r's are in strawberry?"
    ids = tok.encode(sample)
    print(f"\n{sample!r}\n  -> {len(ids)} tokens {ids}\n  -> {tok.pieces(ids)}")
    assert tok.decode(ids) == sample and tok.decode(tok.encode(test)) == test   # lossless: decode(encode(x)) == x
    print(f"chapter XII, never seen: {len(test.encode()):,} bytes -> {len(tok.encode(test)):,} tokens")
