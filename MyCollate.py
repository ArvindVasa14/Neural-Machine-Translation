import torch

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source_vectors = [item[0] for item in batch]
        target_vectors = [item[1] for item in batch]

        source_vectors = self.pad_sequences(source_vectors, self.pad_idx)
        target_vectors = self.pad_sequences(target_vectors, self.pad_idx)

        source_vectors = torch.tensor(source_vectors, dtype=torch.long)
        target_vectors = torch.tensor(target_vectors, dtype=torch.long)

        return source_vectors.to(device), target_vectors.to(device)

    def pad_sequences(self, sequences, pad_idx):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            padded_seq = seq + [pad_idx] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        return padded_sequences

