import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from AttnDecoderRnn import AttnDecoderRnn
from NMT.seq2seq.EncoderRNN import EncoderRNN
from NMT.MyCollate import MyCollate
from NMT.TranslationDataset import TranslationDataset
from tqdm import tqdm

from NMT.Vocabulary import Vocabulary


def get_loader(batch_size, get_check_dataset=True):
    dataset= TranslationDataset("chat", "bot")
    subset= Subset(dataset, list(range(500)))
    dataloader= DataLoader(dataset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)
    sub_dataloader= DataLoader(subset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)

    return (dataloader if get_check_dataset else sub_dataloader,
            dataset if get_check_dataset else subset,
            dataset.source_vocab,
            dataset.target_vocab)


def train_model(encoder, decoder, loader, dataset, source_vocab, target_vocab, epochs, loss_fn, optimizer, save_every= 1):
    batch_size= len(loader)

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch+1} : -----------")
        for source, target in tqdm(loader):
            encoder_outputs, encoder_hidden= encoder(source)
            decoder_outputs= decoder(encoder_outputs, encoder_hidden, target)
            # print(decoder_outputs.size())
            decoder_outputs= decoder_outputs.permute(0, 2, 1)
            # outputs = torch.argmax(decoder_outputs, dim=2)
            loss= loss_fn(decoder_outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+= loss

        print(f"loss: {total_loss/batch_size:.4f}")

        if epoch % save_every==0:
            print("saving weights")
            torch.save(encoder.state_dict(), f"artifacts/models/{dataset.lang1}-{dataset.lang2}_encoder_weights.pth")
            torch.save(decoder.state_dict(), f"artifacts/models/{dataset.lang1}-{dataset.lang2}_decoder_weights.pth")

        random_translate(dataset, encoder, decoder, source_vocab, target_vocab)

def random_translate(dataset, encoder, decoder, source_vocab, target_vocab, n_samples= 5):
    max_len= len(dataset)
    encoder.eval()
    encoder.eval()
    with torch.no_grad():
        for i in range(n_samples):
            n= np.random.randint(0, max_len)
            source, target= dataset[n]
            encoder_outputs, encoder_hidden = encoder(source.unsqueeze(0).to(device))
            decoder_outputs = decoder(encoder_outputs, encoder_hidden, target.unsqueeze(0).to(device))
            outputs = torch.argmax(decoder_outputs, dim=2)
            outputs= outputs.tolist()
            source= source.tolist()
            target= target.tolist()
            for i in range(len(outputs)):
                print(f"\n> : {source_vocab.vector_to_sentence(source)}")
                print(f"< : {target_vocab.vector_to_sentence(target)}")
                print(f"= : {target_vocab.vector_to_sentence(outputs[i])} \n")

    encoder.train()
    decoder.train()

def translate(sentence, source_vocab:Vocabulary, target_vocab:Vocabulary, lang1, lang2):
    encoder = EncoderRNN(embed_size, hidden_size, source_vocab.get_vocab_size()).to(device)
    decoder = AttnDecoderRnn(hidden_size, target_vocab.get_vocab_size()).to(device)

    encoder.load_state_dict(torch.load(f"artifacts/models/{lang1}-{lang2}_encoder_weights.pth"))
    decoder.load_state_dict(torch.load(f"artifacts/models/{lang1}-{lang2}_decoder_weights.pth"))

    sentence= torch.tensor(source_vocab.sentence_to_vector(sentence)).to(device)
    sentence= sentence.unsqueeze(0)
    encoder_output, encoder_hidden= encoder(sentence)
    # print(encoder_output.size(), encoder_hidden.size())
    # encoder_hidden= encoder_hidden.unsqueeze(1)
    decoder_outputs= decoder(encoder_output, encoder_hidden)
    output= torch.argmax(decoder_outputs, dim=2)

    return " ".join(target_vocab.vector_to_sentence(output[0].tolist()))



# Hyperparameters
batch_size= 32
epochs= 50
hidden_size= 512
embed_size= 256
learning_rate= 1e-3

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader, dataset, source_vocab, target_vocab= get_loader(batch_size,True)

encoder= EncoderRNN(embed_size, hidden_size, source_vocab.get_vocab_size()).to(device)
decoder= AttnDecoderRnn( hidden_size, target_vocab.get_vocab_size()).to(device)

loss_fn= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# train_model(encoder, decoder, loader, dataset, source_vocab, target_vocab, epochs, loss_fn, optimizer)

print(translate("how can I go to my home", source_vocab, target_vocab, "chat", "bot"))

























# import torch
# from torch import optim
# import time
# import math
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Subset
#
# from NMT.DecoderRNN import DecoderRNN
# from NMT.EncoderRNN import EncoderRNN
# from NMT.MyCollate import MyCollate
# from NMT.TranslationDataset import TranslationDataset
#
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np
# from torch import nn
#
#
# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)
#
#
# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
#
# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#
#
# def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
#                 decoder_optimizer, criterion):
#     total_loss = 0
#     for data in dataloader:
#         input_tensor, target_tensor = data
#
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#
#         encoder_outputs, encoder_hidden = encoder(input_tensor)
#         decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
#
#         loss = criterion(
#             decoder_outputs.view(-1, decoder_outputs.size(-1)),
#             target_tensor.view(-1)
#         )
#         loss.backward()
#
#         encoder_optimizer.step()
#         decoder_optimizer.step()
#
#         total_loss += loss.item()
#
#     return total_loss / len(dataloader)
#
#
# def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
#           print_every=100, plot_every=100):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every
#
#     encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
#     criterion = nn.NLLLoss()
#
#     for epoch in range(1, n_epochs + 1):
#         loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
#
#         if epoch % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
#                                          epoch, epoch / n_epochs * 100, print_loss_avg))
#
#         if epoch % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0
#
#     showPlot(plot_losses)
#
#
# hidden_size = 512
# batch_size = 32
# embed_size = 256
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# dataset = TranslationDataset("eng", "fra")
# source_vocab, target_vocab = dataset.get_src_and_targ_vocab()
# source_vocab_size = source_vocab.get_vocab_size()
# target_vocab_size = target_vocab.get_vocab_size()
#
# subset = Subset(dataset, list(range(100)))
# train_dataloader = DataLoader(subset, batch_size=32, shuffle=True, drop_last=True, collate_fn=MyCollate(2))
#
# encoder = EncoderRNN(embed_size, hidden_size, source_vocab_size).to(device)
# decoder = DecoderRNN(embed_size, hidden_size, target_vocab_size).to(device)
#
# train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)
