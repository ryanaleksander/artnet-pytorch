import torch

def one_hot_encode(index, num):
    vector = [0 for _ in range(num)]
    vector[index] = 1
    return torch.Tensor(vector)