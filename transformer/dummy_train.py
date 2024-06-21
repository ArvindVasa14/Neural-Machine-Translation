import random

import numpy as np
import torch

def generate_random_data(n):
    SOS_token= np.array([2])
    EOS_token= np.array([3])
    length= 8

    data= []

    for i in range(n//3):
        X= np.concatenate((SOS_token, np.ones(length), EOS_token))
        y= np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    for i in range(n//3):
        X= np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y= np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])


    for i in range(n//3):
        X= np.zeros(length)
        start= random.randint(0,1)

        X[start::2]=1

        y= np.zeros(length)
        if X[-1]==0:
            y[::2]=1
        else:
            y[1::2]=1

        X= np.concatenate((SOS_token, X, EOS_token))
        y= np.concatenate((SOS_token, y, EOS_token))

        data.append([X, y])

    np.random.shuffle(data)

    return np.array(data)


def batchify_data(data, batch_size= 16, padding= False, padding_token= -1):
    batches= []
    for idx in range(0, len(data), batch_size):
        if idx+batch_size<len(data):
            if padding:
                max_batch_len= 0

                for seq in data[idx: idx+batch_size]:
                    if len(seq) > max_batch_len:
                        max_batch_len= len(seq)

                for seq_idx in range(batch_size):
                    remaining_length= max_batch_len - len(data[idx + seq_idx])
                    data[idx + seq_idx ] += [padding_token] * remaining_length

            batches.append(np.array(data[idx:idx+batch_size]).astype(np.int64))

    return np.array(batches)



if __name__=="__main__":
    data= generate_random_data(100)
    print(data.shape)
    print(batchify_data(data).shape)