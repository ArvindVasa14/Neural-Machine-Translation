non_english_languages= ["jap"]
class Vocabulary:
    def __init__(self, lang_name, specials):
        self.word2idx= {}
        self.idx2word= {}
        self.wordscount= {}
        self.total_words= 0
        self.lang_name= lang_name

        for special in specials:
            self.word2idx[special]= self.total_words
            self.idx2word[self.total_words]= special
            self.total_words+=1
            self.wordscount[special]= 1

    def add_sentence(self, sentence):
        sentence= sentence.lower().strip().split(" ") if self.lang_name not in non_english_languages else list(sentence)
        for word in sentence:
            if word not in self.word2idx:
                word= word.strip()
                self.word2idx[word] = self.total_words
                self.idx2word[self.total_words] = word
                self.total_words += 1
                self.wordscount[word]= 1
            else:
                self.wordscount[word]+=1

    def get_vocab_size(self):
        return self.total_words

    def sentence_to_vector(self, sentence):
        if self.lang_name not in non_english_languages:
            return [self.word2idx[word] if word in self.word2idx else self.word2idx["<UNK>"] for word in sentence.lower().strip().split(" ")]
        else:
            return [self.word2idx[word] if word in self.word2idx else self.word2idx["<UNK>"] for word in list(sentence)]

    def vector_to_sentence(self, vector):
        return " ".join([self.idx2word[idx] if idx in self.idx2word else self.idx2word[3] for idx in vector])



# vocab= Vocabulary("english", ["<SOS>", "<EOS>"])
# vocab.add_sentence("hi I am arvind")
# print(vocab.get_vocab_size())
# print(vocab.word2idx, vocab.idx2word)

# print(vocab.sentence_to_vector("hi I am arvind"))
# print(vocab.vector_to_sentence([2, 3, 4, 5]))