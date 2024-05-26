import torch


def b_n_hd_2_bh_n_d(t, h):
    b, n, hd = t.shape
    d = hd // h
    t = t.view(b, n, h, d)
    t = t.permute(0, 2, 1, 3).contiguous()
    t = t.view(b * h, n, d)
    return t


def bh_n_d_2_b_n_hd(t, h):
    bh, n, d = t.shape
    b = bh // h
    t = t.view(b, h, n, d)
    t = t.permute(0, 2, 1, 3).contiguous()
    t = t.view(b, n, h * d)
    return t


def b_multi_2_b_1(t):
    b = t.shape[0]
    return t.view(b, -1)


def repeat_with_h(t, h):
    b, j = t.shape
    t = t.unsqueeze(1).repeat(1, h, 1).view(b * h, 1, j)
    return t
