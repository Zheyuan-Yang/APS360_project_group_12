import torch


def encoding_1d(pos, dim):
    vec = torch.arange(pos)

    # if dim % 2 != 0:
    #     return

    emb = torch.zeros(vec.numel(), dim, dtype=torch.float)

    itr = torch.arange(dim/2, dtype= torch.float)
    itr /= dim/2.
    itr = 1. / (10000 ** itr)

    out = vec[:, None] @ itr[None,:]

    eSin = torch.sin(out)
    eCos = torch.cos(out)

    emb[:, 0::2] = eSin
    emb[:, 1::2] = eCos

    return emb
