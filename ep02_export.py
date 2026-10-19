"""Every experiment in Ep 2 "Tokens", saved to runs/ep02.json. Every number in the video comes from this file.

    python3 -m venv .venv && .venv/bin/pip install tiktoken
    .venv/bin/python ep02_export.py         # about 30 s
"""
import json, platform, time
from collections import Counter
import tiktoken
from bpe import BPE, SPLIT, count_pairs

R = {"machine": platform.machine(), "python": platform.python_version(), "tiktoken": tiktoken.__version__,
     "date": time.strftime("%Y-%m-%d")}
real = {n: tiktoken.get_encoding(n) for n in ["gpt2", "cl100k_base", "o200k_base", "o200k_harmony"]}
piece = lambda enc, i: enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
split = lambda enc, s: {"ids": enc.encode(s), "pieces": [piece(enc, i) for i in enc.encode(s)]}

# 1. the fruit stand: BPE by hand on four words ------------------------------------------------------------
toy = "strawberry blueberry raspberry blackberry"
chunks = Counter(SPLIT.findall(toy))
words = {c: list(c.encode()) for c in chunks}
steps, tok = [], BPE()
for new_id in range(256, 256 + 8):
    counts = Counter()
    for c, ids in words.items():
        count_pairs(ids, counts, chunks[c])
    show = lambda i: tok.vocab[i].decode()
    table = [[show(a), show(b), n] for (a, b), n in counts.most_common()]
    best = max(counts, key=counts.get)
    tok.merges[best] = new_id
    tok.vocab[new_id] = tok.vocab[best[0]] + tok.vocab[best[1]]
    words = {c: tok.encode(c) for c in words}
    steps.append({"merge": new_id - 255, "pair": [show(best[0]), show(best[1])], "count": counts[best], "new_id": new_id,
                  "token": show(new_id), "pair_counts": table, "after": {c: [show(i) for i in ids] for c, ids in words.items()}})
R["toy"] = {"text": toy, "steps": steps}

# 2. a real book: Alice's Adventures in Wonderland, chapters I-XI to learn from, chapter XII kept unseen ----
book = open("data/alice.txt", encoding="utf-8").read()
cut = book.index("CHAPTER XII.")
train, test = book[:cut], book[cut:]
log = []
t0 = time.time()
alice = BPE().train(train, 256 + 4000, lambda i, p, n, b: log.append([i - 255, i, b.decode("utf-8", errors="replace"), n, list(p)]))
t_4000 = time.time() - t0
t0 = time.time(); BPE().train(train, 1000); t_744 = time.time() - t0
named = [" the", " and", " said", "Alice", " Alice", " little", " thought", " Queen", " Hatter", " Mock", " Turtle", " Cat",
         " Gryphon", " Dormouse", " King", " Duchess", " Rabbit", " March", " Hare", " Caterpillar", " Mouse", " curious", " Cheshire"]
first = {w: next(([m, n] for m, i, s, n, p in log if s == w), None) for w in named}


def truncated(tok, vocab_size):                            # the same tokenizer, stopped after fewer merges
    t = BPE()
    t.merges = {p: i for p, i in tok.merges.items() if i < vocab_size}
    t.vocab = {i: b for i, b in tok.vocab.items() if i < vocab_size}
    return t


sizes = [256, 300, 500, 1000, 2000, 4256]
curve = []
for v in sizes:
    t = truncated(alice, v)
    n = len(t.encode(test))
    curve.append({"vocab": v, "merges": v - 256, "tokens": n, "bytes_per_token": round(len(test.encode()) / n, 3),
                  "chars_per_token": round(len(test) / n, 3), "strawberry": t.pieces(t.encode(" strawberry")),
                  "first_line": t.pieces(t.encode("Alice was beginning to get very tired of sitting by her sister"))})
a1000 = truncated(alice, 1000)
assert a1000.decode(a1000.encode(book)) == book and alice.decode(alice.encode(test)) == test   # lossless
q = "How many r's are in strawberry?"
R["alice"] = {"chars": len(book), "words": len(book.split()), "bytes": len(book.encode()),
              "train_chars": len(train), "train_bytes": len(train.encode()), "test_chars": len(test), "test_bytes": len(test.encode()),
              "test_words": len(test.split()), "seconds_744_merges": round(t_744, 2), "seconds_4000_merges": round(t_4000, 2),
              "first_merges": log[:40], "every_100th": [m for m in log if m[0] % 100 == 0], "named": first, "curve": curve,
              "question_1000": {"ids": a1000.encode(q), "pieces": a1000.pieces(a1000.encode(q))},
              "merge_3_bytes": list(alice.vocab[258]), "curly_quotes": {c: list(c.encode()) for c in "“”’"}}

# 3. special tokens in our own tokenizer, and a chat template --------------------------------------------
a1000.add_special("<|user|>", "<|assistant|>", "<|end|>")
chat = "<|user|>How many r's are in strawberry?<|end|><|assistant|>"
R["ours_special"] = {"ids": a1000.special, "chat": a1000.encode(chat, allow_special=True),
                     "chat_pieces": a1000.pieces(a1000.encode(chat, allow_special=True)),
                     "typed_end": a1000.pieces(a1000.encode("<|end|>"))}

# 4. real tokenizers -------------------------------------------------------------------------------------
o200k, harmony = real["o200k_base"], real["o200k_harmony"]
variants = ["strawberry", " strawberry", "Strawberry", " Strawberry", "STRAWBERRY", q, "s t r a w b e r r y", "s-t-r-a-w-b-e-r-r-y",
            " raspberry", " blueberry", " SolidGoldMagikarp", "12345678", " unbelievable", " tokenization", "🍓"]
R["real"] = {n: {"n_vocab": e.n_vocab, "specials": {s: e.encode_single_token(s) for s in sorted(e.special_tokens_set) if "reserved" not in s},
                 "split": {s: split(e, s) for s in variants}} for n, e in real.items()}
R["real"]["o200k_base"]["r_id"] = o200k.encode(" r")[0]
R["real"]["o200k_base"]["spelled_r_count"] = o200k.encode("s t r a w b e r r y").count(o200k.encode(" r")[0])
convo = "<|start|>user<|message|>How many r's are in strawberry?<|end|><|start|>assistant"
ids = harmony.encode(convo, allowed_special="all")
try:
    harmony.encode("<|end|>"); typed = "no error"
except ValueError as e:
    typed = "ValueError: " + str(e).split("\n")[0]
as_text = harmony.encode("<|end|>", disallowed_special=())     # what a user who types "<|end|>" should become
R["harmony"] = {"text": convo, "ids": ids, "pieces": [piece(harmony, i) for i in ids], "typed_end_default": typed,
                "typed_end_as_text": {"ids": as_text, "pieces": [piece(harmony, i) for i in as_text]}}
R["books"] = {n: {"tokens": len(e.encode(book, disallowed_special=())), "test_tokens": len(e.encode(test, disallowed_special=()))}
              for n, e in real.items() if n != "o200k_harmony"}
R["books"]["ours_1000"] = {"tokens": len(a1000.encode(book)), "test_tokens": len(a1000.encode(test))}
R["books"]["bytes"] = {"tokens": len(book.encode()), "test_tokens": len(test.encode())}
for n, b in R["books"].items():
    b["chars_per_token"] = round(len(book) / b["tokens"], 2)
    b["words_per_token"] = round(len(book.split()) / b["tokens"], 3)

# 5. the token tax: one sentence (UDHR Article 1) in 21 languages ------------------------------------------
udhr = json.load(open("data/udhr_article1.json", encoding="utf-8"))["articles"]
tax = {}
for code, a in udhr.items():
    s = a["text"]
    tax[code] = {"name": a["name"], "chars": len(s), "bytes": len(s.encode()), "ours_1000": len(a1000.encode(s)),
                 **{n: len(e.encode(s)) for n, e in real.items() if n != "o200k_harmony"}}
for code in tax:
    for k in ["bytes", "ours_1000", "gpt2", "cl100k_base", "o200k_base"]:
        tax[code][k + "_x"] = round(tax[code][k] / tax["eng"][k], 2)
R["tax"] = tax
R["tax_examples"] = {c: split(o200k, udhr[c]["text"]) for c in ["eng", "spa", "hin", "mya", "jpn"]}

# 6. UTF-8: how many bytes one character takes ------------------------------------------------------------
R["utf8"] = {ch: list(ch.encode("utf-8")) for ch in ["a", "é", "म", "က", "🍓"]}
R["spelled"] = {s: split(o200k, s) for s in ["s t r a w b e r r y", "s-t-r-a-w-b-e-r-r-y"]}
R["cafe"] = {"chars": len("café"), "js_length": len("café".encode("utf-16-le")) // 2, "bytes": list("café".encode()),
             "byte_tokens": len("café".encode()), "o200k": split(o200k, "café")}
R["emoji"] = {"js_length": len("🍓".encode("utf-16-le")) // 2, "bytes": len("🍓".encode()), "o200k_tokens": len(o200k.encode("🍓"))}
# our book-trained tokenizer on English that isn't Wonderland (UDHR Article 1), against GPT-2
a4256 = truncated(alice, 4256)
R["other_english"] = {"ours_4256": len(a4256.encode(udhr["eng"]["text"])), "ours_1000": len(a1000.encode(udhr["eng"]["text"])),
                      "gpt2": len(real["gpt2"].encode(udhr["eng"]["text"])), "o200k": len(o200k.encode(udhr["eng"]["text"]))}

json.dump(R, open("runs/ep02.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"toy: {[s['token'] for s in steps]}")
print(f"alice: 744 merges {t_744:.1f} s, 4000 merges {t_4000:.1f} s; named {first}")
print("curve:", [(c["vocab"], c["tokens"], c["chars_per_token"]) for c in curve])
print("books:", {n: (b["tokens"], b["chars_per_token"]) for n, b in R["books"].items()})
print("tax o200k x:", {t["name"]: t["o200k_base_x"] for t in tax.values()})
print("wrote runs/ep02.json")
