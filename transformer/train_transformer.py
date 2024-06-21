import torch
from torch import nn

from NMT.transformer.TransformerModel import Transformer
from NMT.transformer.dummy_train import generate_random_data, batchify_data

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def train_loop(model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss= 0

    for batch in tqdm(dataloader):
        X, y= batch[:,0], batch[:, 1]
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

    with torch.no_grad():
        for batch in tqdm(dataloader):
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            # if sentence is : <SOS> Hi I am Groot <EOS>
            # y_inpit : <SOS> Hi I am Groot
            # y_expected : Hi I am Groot <EOS>

            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(X, y_input, tgt_mask)

            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list= [], []

    print("Training and Validating model")

    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch+1}", "-"*25)

        train_loss= train_loop(model, optimizer, loss_fn, train_dataloader)
        train_loss_list+= [train_loss]

        validation_loss= validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list+= [validation_loss]

        print(f"Training loss : {train_loss:.4f}")
        print(f"Validation loss : {validation_loss:.4f}")

    return train_loss_list, validation_loss_list



device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Transformer(
    num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)

optimizer= torch.optim.SGD(model.parameters(), lr= 0.01)
loss_fn= nn.CrossEntropyLoss()

train_data= generate_random_data(9000)
val_data= generate_random_data(3000)

train_dataloader= batchify_data(train_data)
val_dataloader= batchify_data(val_data)

train_loss_list, validation_loss_list= fit(model, optimizer, loss_fn, train_dataloader, val_dataloader, 10)