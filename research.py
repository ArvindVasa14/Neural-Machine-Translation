import torch
import unicodedata

import re

def unicodeToAscii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
    )

sentence= "周りの人に聞いてみて。"
print(sentence)

sentence = unicodeToAscii(sentence.lower().strip())
sentence = re.sub(r"([.!?])", r" \1", sentence)
# sentence = re.sub(r"[^a-zA-Z!?]+", r" ", sentence)

print(list(sentence))