import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from NMT.MyCollate import MyCollate
from NMT.transformer.TranslationDataset import TranslationDataset
from NMT.transformer.TransformerModel import Transformer
from NMT.transformer.dummy_train import generate_random_data, batchify_data

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import numpy as np
import pickle
import sacrebleu

def train_loop(model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss= 0

    for source, target in tqdm(dataloader):
        X, y= source, target
        X, y= torch.tensor(X).to(device), torch.tensor(y).to(device)

        # if sentence is : <SOS> Hi I am Groot <EOS>
        # y_inpit : <SOS> Hi I am Groot
        # y_expected : Hi I am Groot <EOS>

        y_input= y[:, :-1]
        y_expected= y[:, 1:]
        sequence_length= y_input.size(1)
        tgt_mask= model.get_tgt_mask(sequence_length).to(device)

        pred= model(X, y_input, tgt_mask)

        pred= pred.permute(1, 2, 0)
        loss= loss_fn(pred, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.detach().item()

    return total_loss/ len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss= 0

    actuals= []
    predicted= []

    with torch.no_grad():
        for source, target in tqdm(dataloader):
            X, y = source, target
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            # if sentence is : <SOS> Hi I am Groot <EOS>
            # y_inpit : <SOS> Hi I am Groot
            # y_expected : Hi I am Groot <EOS>

            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)

            batch_pred= [target_vocab.vector_to_sentence(p) for p in pred]

            actuals.extend([target_vocab.vector_to_sentence(a) for a in y])
            predicted.extend(batch_pred)

            pred = pred.permute(1, 2, 0)

            loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()
    bleu= sacrebleu.corpus_bleu(predicted, actuals)
    return total_loss / len(dataloader) , bleu

def fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs, save_every= 2):
    train_loss_list, validation_loss_list= [], []

    print("Training and Validating model")

    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch+1}", "-"*25)

        train_loss= train_loop(model, optimizer, loss_fn, train_dataloader)
        train_loss_list+= [train_loss]

        validation_loss, bleu= validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list+= [validation_loss]

        print(f"Training loss : {train_loss:.4f}")
        print(f"Validation loss : {validation_loss:.4f}")
        print(f"Bleu score : {bleu}")

        for k in range(5):
            n = np.random.randint(0, len(dataset))
            input_sequence, target_sequence = dataset[n]
            print(f"> {source_vocab.vector_to_sentence(input_sequence.tolist())}")
            print(f"< {target_vocab.vector_to_sentence(target_sequence.tolist())}")
            print(f"= {target_vocab.vector_to_sentence(predict(model, input_sequence.unsqueeze(0)))} \n")

        model.train()

        if epoch % save_every==0:
            print("saving checkpoint..")
            torch.save(model.state_dict(), "models/eng-jap_translation_model_weights.pth")

    return train_loss_list, validation_loss_list

def get_loader(batch_size, get_check_dataset=True):
    train_dataset= TranslationDataset("eng", "jap", "data/train_eng-jap.txt", False)
    test_dataset= TranslationDataset("eng", "jap", "data/test_eng-jap.txt")

    train_subset= Subset(train_dataset, list(range(500)))
    test_subset= Subset(test_dataset, list(range(500)))

    train_dataloader= DataLoader(train_dataset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)
    test_dataloader= DataLoader(test_dataset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)

    train_sub_dataloader= DataLoader(train_subset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)
    test_sub_dataloader= DataLoader(test_subset, batch_size= batch_size, shuffle= True, collate_fn= MyCollate(2), drop_last= True)

    return (train_dataloader if get_check_dataset else train_sub_dataloader,
            test_dataloader if get_check_dataset else test_sub_dataloader,
            train_dataset if get_check_dataset else train_subset,
            train_dataset.source_vocab,
            train_dataset.target_vocab)

def predict(model, input_sequence, max_length= 15, SOS_token=0, EOS_token= 1):
    model.eval()
    input_sequence= input_sequence.to(device)
    y_input= torch.tensor([[SOS_token]], dtype= torch.long, device= device)
    num_tokens= len(input_sequence[0])

    for _ in range(max_length):
        tgt_mask= model.get_tgt_mask(y_input.size(1)).to(device)

        pred= model(input_sequence, y_input, tgt_mask)

        next_item= pred.topk(1)[1].view(-1)[-1].item()
        next_item= torch.tensor([[next_item]], device= device)

        y_input= torch.cat((y_input, next_item), dim=1)

        if next_item.view(-1).item()==EOS_token:
            break

    model.train()

    return y_input.view(-1).tolist()


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device("cpu")

train_dataloader, val_dataloader, dataset,  source_vocab, target_vocab= get_loader(32, True)

with open("artifacts/source_vocab.pkl","wb") as file:
    pickle.dump(source_vocab, file)

with open("artifacts/target_vocab.pkl","wb") as file:
    pickle.dump(target_vocab, file)


model= Transformer(
    src_vocab_size=source_vocab.get_vocab_size(), tgt_vocab_size= target_vocab.get_vocab_size(), dim_model= 256, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)

optimizer= torch.optim.SGD(model.parameters(), lr= 0.01)
loss_fn= nn.CrossEntropyLoss()

# train_data= generate_random_data(9000)
# val_data= generate_random_data(3000)
#
# train_dataloader= batchify_data(train_data)
# val_dataloader= batchify_data(val_data)


train_loss_list, validation_loss_list= fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, 5, 1)