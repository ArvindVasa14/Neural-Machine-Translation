import torch
import unicodedata
from torch.utils.data import Dataset, DataLoader, Subset
import re
import os
from NMT.MyCollate import MyCollate
from NMT.Vocabulary import Vocabulary
import pickle

non_english_languages= ["jap"]

class TranslationDataset(Dataset):
    def __init__(self, lang1, lang2, reuse_vocabulary= True):
        self.lang1= lang1
        self.lang2= lang2
        self.lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
        if self.lang1 not in non_english_languages and self.lang2 not in non_english_languages:
            self.lines= [[self.clean_sentence(s) for s in line.split("\t")] for line in self.lines]
        else:
            self.lines= [[self.clean_sentence(s, True) for s in line.split("\t")] for line in self.lines]


        if os.path.isfile(f'artifacts/{lang1}.pkl') and os.path.isfile(f'artifacts/{lang2}.pkl') and reuse_vocabulary:
            print("Using existing vocab files")
            with open(f'artifacts/{lang1}.pkl', 'rb') as source_vocab_file, open(f'artifacts/{lang2}.pkl', 'rb') as target_vocab_file:
                self.source_vocab = pickle.load(source_vocab_file)
                self.target_vocab = pickle.load(target_vocab_file)

        else:
            print("Generating vocab files")
            self.source_vocab= Vocabulary(self.lang1, ["<SOS>", "<EOS>","<PAD>","<UNK>"])
            self.target_vocab = Vocabulary(self.lang2, ["<SOS>", "<EOS>","<PAD>","<UNK>"])

            for src_sent, targ_sent in self.lines:
                self.source_vocab.add_sentence(src_sent)
                self.target_vocab.add_sentence(targ_sent)

            with open(f'artifacts/{lang1}.pkl', 'wb') as source_vocab_file, open(f'artifacts/{lang2}.pkl', 'wb') as target_vocab_file:
                pickle.dump(self.source_vocab, source_vocab_file)
                pickle.dump(self.target_vocab, target_vocab_file)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        pair= self.lines[idx]
        return (torch.tensor(self.source_vocab.sentence_to_vector(pair[0])),
                torch.tensor(self.target_vocab.sentence_to_vector(pair[1])))

    def get_src_and_targ_vocab(self):
        return self.source_vocab, self.target_vocab

    def unicodeToAscii(self, sentence):
        return ''.join(
            c for c in unicodedata.normalize('NFD', sentence)
            if unicodedata.category(c) != 'Mn'
        )

    def clean_sentence(self, sentence, non_english= False):
        sentence = self.unicodeToAscii(sentence.lower().strip())
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z!?]+", r" ", sentence) if not non_english else sentence
        return sentence.strip()


if __name__=="__main__":
    dt= TranslationDataset("eng","jap", False)
    # loader= DataLoader(dt, batch_size=4, shuffle=True, drop_last= True, collate_fn= MyCollate(2))
    d= dt[100000]
    print(d)
    print(dt.target_vocab.get_vocab_size())
    print("=", dt.target_vocab.vector_to_sentence(d[1].tolist()))