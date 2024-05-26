from inspect import isfunction
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from einops_exts import check_shape, rearrange_many
from .rearrange import *
from rotary_embedding_torch import RotaryEmbedding

try:
    import os
    import xformers
    import xformers.ops

    print(xformers)
    XFORMERS_IS_AVAILBLE = bool(int(os.environ.get("USE_XFORMERS", True)))
except:
    XFORMERS_IS_AVAILBLE = False

import numpy as np

import pdb
bp=pdb.set_trace


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    return func(*inputs)
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
    """


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        dim = len(x.shape) - 1
        x, gate = self.proj(x).chunk(2, dim=dim)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        # q = rearrange(q, 'b c h w -> b (h w) c')
        # k = rearrange(k, 'b c h w -> b c (h w)')
        q = q.view(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.view(b, c, h * w)

        # w_ = torch.einsum('bij,bjk->bik', q, k)
        w_ = torch.matmul(q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        # v = rearrange(v, 'b c h w -> b c (h w)')
        # w_ = rearrange(w_, 'b i j -> b j i')
        b, c, h, w = v.shape
        v = v.view(b, c, h * w)
        w_ = w_.permute(0, 2, 1).contiguous()

        # h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = torch.matmul(v, w_)

        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_pvt_sa=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        # import pdb; pdb.set_trace()
        context_dim = default(context_dim, query_dim)
        self.use_pvt_sa = use_pvt_sa

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        # import pdb; pdb.set_trace()
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q, k, v = map(lambda t: b_n_hd_2_bh_n_d(t, h=h), (q, k, v))

        if self.use_pvt_sa:
            height = width = int(np.sqrt(q.shape[1]))
            assert (
                height**2 == q.shape[1]
            ), "[CrossAttentionWithDownsize] => currently downsize does not support non-square image! Your (h w) shape: {}".format(
                q.shape[1]
            )
            import pdb

            pdb.set_trace()

            v, k = map(
                lambda t: rearrange(
                    t, "(b a) (h w) d -> (b a) d h w", a=h, h=height, w=width
                ),
                (v, k),
            )  # a: head
            if 1:
                v = F.interpolate(v, height // 4, mode="nearest")
                k = F.interpolate(k, height // 4, mode="nearest")
            else:
                pool = nn.AdaptiveAvgPool2d((height // 4, width // 4))
                v = pool(v)
                k = pool(k)
            v, k = map(
                lambda t: rearrange(
                    t, "(b a) d h w -> (b a) (h w) d", a=h, h=height // 4, w=width // 4
                ),
                (v, k),
            )  # a: head

        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.matmul(q, k.permute(0, 2, 1).contiguous()) * self.scale

        if exists(mask):
            # mask = rearrange(mask, 'b ... -> b (...)')
            mask = b_multi_2_b_1(mask)

            max_neg_value = -torch.finfo(sim.dtype).max
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            mask = repeat_with_h(mask, h)

            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=2)

        # out = einsum('b i j, b j d -> b i d', attn, v)
        out = torch.matmul(attn, v)

        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = bh_n_d_2_b_n_hd(out, h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.with_ip = kwargs.get("with_ip", False)

        # if context_dim is not None:
        #     bp()
        if self.with_ip and (context_dim is not None):
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.ip_dim= kwargs.get("ip_dim", 16)
            self.ip_weight = kwargs.get("ip_weight", 1.0)
            # self.ip_weight = kwargs.get("ip_weight", 0.0)
            # print(self.ip_weight)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)

        has_ip = self.with_ip and (context is not None)
        if has_ip:
            # context dim [b frame_num 77 + 4 1024 ]
            token_len = context.shape[1]
            context_ip = context[:, -self.ip_dim:, :]
            k_ip = self.to_k_ip(context_ip)
            v_ip = self.to_v_ip(context_ip)
            context = context[:, :(token_len - self.ip_dim), :]

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if has_ip:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            # actually compute the attention, what we cannot get enough of
            out_ip = xformers.ops.memory_efficient_attention(
                q, k_ip, v_ip, attn_bias=None, op=self.attention_op
            )
            out = out + self.ip_weight * out_ip

        if exists(mask):
            raise NotImplementedError

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


def log_weight_t(weight_t, prob=0.1):
    if (
        torch.rand(
            1,
        )
        < prob
    ):
        _weight_t_log = weight_t.view(-1, 5, 5).mean(0).flatten()
        _log = "".join(
            ["{:.4f},".format(i) for i in _weight_t_log.detach().cpu().numpy()]
        )
        with open("./_attention_log.csv", "a") as fout:
            fout.write("weight_t_softmax_mean,{}\n".format(_log))


class CrossAttentionTemporal(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attend_prev_only=True,
        mask_mode="v",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attend_prev_only = attend_prev_only

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        pos_bias=None,
        focus_present_mask_only=False,
        t_temp=1.0,
        diag_offset=1,
    ):
        # b (h w) 5 c
        h = self.heads
        num_frames = x.shape[2]
        s = x.shape[1]
        bs = x.shape[0]
        device = x.device

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = rearrange_many(
            (q, k, v), "b s n (h d) -> (b s) h d n", h=h, n=num_frames, s=s
        )  #  (bhw) heads, dim_head, 5

        if rotary_emb is not None:
            q = rearrange(q, "b h c f -> b h f c")
            k = rearrange(k, "b h c f -> b h f c")

            q = q * self.scale
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

            q = rearrange(q, "b h f c -> b h c f")
            k = rearrange(k, "b h f c -> b h c f")

            weight_t = torch.einsum(
                "bhct,bhcs->bhts", q, k
            )  # More stable with f16 than dividing afterwards, bhw, n_heads, 5, 5

        else:
            weight_t = torch.einsum(
                "bhct,bhcs->bhts", q * self.scale, k
            )  # More stable with f16 than dividing afterwards, bhw, n_heads, 5, 5

        if pos_bias is not None:
            # import pdb; pdb.set_trace()
            weight_t = weight_t + pos_bias  # bhw, n_heads, 5, 5

        if self.attend_prev_only:
            mask_triu = torch.ones(
                (num_frames, num_frames), device=device, dtype=torch.float32
            )
            # import pdb; pdb.set_trace()
            mask_triu = torch.triu(mask_triu, diagonal=diag_offset)
            mask_triu = mask_triu.masked_fill(mask_triu == 1, float("-inf"))
            weight_t = weight_t + rearrange(mask_triu, "i j -> 1 1 i j")

        # import pdb; pdb.set_trace()
        if focus_present_mask_only:
            mask_eye = 1 - torch.eye(
                num_frames, num_frames, device=device, dtype=torch.float32
            )
            mask_eye = mask_eye.masked_fill(mask_eye == 1, float("-inf"))
            weight_t = weight_t + rearrange(mask_eye, "i j -> 1 1 i j")

        weight_t = weight_t - weight_t.amax(dim=-1, keepdim=True).detach()

        weight_t = torch.softmax(weight_t.float() / t_temp, dim=-1).type(weight_t.dtype)
        # import pdb; pdb.set_trace()

        # log_weight_t(weight_t, prob=0.1)

        res = torch.einsum("bhts,bhcs->bhct", weight_t, v).reshape(
            bs * s, -1, num_frames
        )  # (b h*w), inner_dim, 5

        res = rearrange(res, "(b s) c f -> b s f c", s=s)
        # import pdb; pdb.set_trace()

        return self.to_out(res)

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CrossAttentionTemporal_xformer(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attend_prev_only=False,
        mask_mode="v",
        temporal_position_encoding_max_len = 24,
        use_pose_embed=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attend_prev_only = attend_prev_only
        self.use_pose_embed = use_pose_embed

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

        self.mask_mode = mask_mode

        # self.pos_encoder = PositionalEncoding(
        #     query_dim,
        #     dropout=0.,
        #     max_len=temporal_position_encoding_max_len
        # )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        pos_bias=None,
        focus_present_mask_only=False,
        t_temp=1.0,
        diag_offset=1,
    ):
        # b (h w) 5 c
        h = self.heads
        num_frames = x.shape[2]
        s = x.shape[1]
        bs = x.shape[0]
        device = x.device

        # if self.use_pose_embed:
        #     x = self.pos_encoder(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if focus_present_mask_only:
            if self.mask_mode == "2d":
                # Equivalent to MemoryEfficientCrossAttention of 2D Version
                q, k, v = rearrange_many(
                    (q, k, v), "b s f (h c) -> (b f h) s c",
                    h=h,
                    f=num_frames,
                    s=s
                )  #  (bhw) heads, dim_head, 5
                res = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )
                res = rearrange(
                    res, "(b f h) s c -> b s f (h c)", h=h, f=num_frames, s=s
                )  #  (bhw) heads, dim_head, 5
            elif self.mask_mode == "v":
                res = v
            else:
                raise NotImplemented(f"unknown mask mode {mask_mode}")
        else:
            q, k, v = rearrange_many(
                (q, k, v), "b s f (h c) -> (b s h) f c", h=h, f=num_frames, s=s
            )  #  (bhw heads), f, dim_head

            if rotary_emb is not None:
                assert False

            res = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

            res = rearrange(res, "(b s h) f c -> b s f (h c)", h=h, f=num_frames, s=s)

        return self.to_out(res)

class CrossAttention_plain(nn.Module):
    """Similar to CrossAttentionTemporal but fully attended"""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attend_prev_only=True,
        mask_mode="v",
        with_ip=False,
        ip_len=4,
        ip_weight=1.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attend_prev_only = attend_prev_only

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None
        self.mask_mode = mask_mode

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        pos_bias=None,
        focus_present_mask_only=False,
        t_temp=1.0,
        diag_offset=1,
    ):
        # b (h w) 5 c
        h = self.heads
        num_frames = x.shape[2]
        s = x.shape[1]
        bs = x.shape[0]
        device = x.device

        q = self.to_q(x)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Equivalent to MemoryEfficientCrossAttention of 2D Version
        q, k, v = rearrange_many(
            (q, k, v), "b s f (h c) -> (b f h) s c",
            h=h,
            f=num_frames,
            s=s
        )  #  (bhw) heads, dim_head, 5
        res = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )
        res = rearrange(
            res, "(b f h) s c -> b s f (h c)", h=h, f=num_frames, s=s
        )  #  (bhw) heads, dim_head, 5

        return self.to_out(res)


class CrossAttentionTemporal3D(nn.Module):
    """Similar to CrossAttentionTemporal but fully attended"""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attend_prev_only=True,
        mask_mode="v",
        with_ip=False,
        ip_len=4,
        ip_weight=1.0,
        use_sep_selfattn=False,
        train_4viewip=False,
        anchor_infer_once=False,
        random_drop_rate=0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attend_prev_only = attend_prev_only

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.with_ip = with_ip
        if with_ip and context_dim:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.ip_len = ip_len
            self.ip_weight = ip_weight

        self.train_4viewip = train_4viewip
        self.anchor_infer_once = anchor_infer_once
        if self.anchor_infer_once:
            self.anchor_k = None
            self.anchor_v = None
            self.frame_num = None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None
        self.mask_mode = mask_mode
        self.use_sep_selfattn = use_sep_selfattn
        self.random_drop_rate = random_drop_rate

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        pos_bias=None,
        focus_present_mask_only=False,
        t_temp=1.0,
        diag_offset=1,
        mode=3,
        attn_mode='org',
        sliding_wondow=None,
        reset=False,
        drop_mask=None,
    ):
        # b (h w) 5 c
        h = self.heads
        num_frames = x.shape[2]
        s = x.shape[1]
        bs = x.shape[0]
        device = x.device

        q = self.to_q(x)

        if self.with_ip and context:
            # context dim [b frame_num 77 + 4 1024 ]
            token_len = context.shape[2]
            context_ip = context[:, :, -self.ip_len:, :]
            k_ip = self.to_k_ip(context_ip)
            v_ip = self.to_v_ip(context_ip)
            context = context[:, :, :(token_len - self.ip_len), :]

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        first = False
        if self.anchor_infer_once and (self.anchor_k is None or reset):
            can_k = k[:, :, 1:].detach()
            self.anchor_k = can_k
            can_v = v[:, :, 1:].detach()
            self.anchor_v = can_v
            self.frame_num = k.shape[2]
            first = True
        elif self.anchor_infer_once and self.anchor_k is not None:
            k = torch.cat([k, self.anchor_k], dim=2)
            v = torch.cat([v, self.anchor_v], dim=2)
            num_frames = self.frame_num

        mask = None
        if focus_present_mask_only:
            if self.mask_mode == "2d":
                # Equivalent to MemoryEfficientCrossAttention of 2D Version
                q, k, v = rearrange_many(
                    (q, k, v), "b s f (h c) -> (b f h) s c",
                    h=h,
                    f=num_frames,
                    s=s
                )  #  (bhw) heads, dim_head, 5
                res = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )
                res = rearrange(
                    res, "(b f h) s c -> b s f (h c)", h=h, f=num_frames, s=s
                )  #  (bhw) heads, dim_head, 5
            elif self.mask_mode == "v":
                res = v
            else:
                raise NotImplemented(f"unknown mask mode {mask_mode}")
        else:
            if self.use_sep_selfattn:
                if not self.anchor_infer_once or first:
                    qs, ks, vs = q[:, :, 1:], k[:, :, 1:], v[:, :, 1:]
                    qs, ks, vs = rearrange_many(
                        (qs, ks, vs), "b s f (h c) -> (b f h) s c",
                        h=h,
                        f=num_frames - 1,
                        s=s
                    )

                    ress = xformers.ops.memory_efficient_attention(
                        qs, ks, vs, attn_bias=None, op=self.attention_op
                    )
                    ress = rearrange(
                        ress, "(b f h) s c -> b s f (h c)", h=h, f=num_frames - 1, s=s
                    )

                qt = q[:, :, :1]
                qt = rearrange(
                    qt, "b s f (h c) -> (b h) (s f) c", h=h, f=1, s=s
                )

                if attn_mode == 'org':
                    if mode == 3:

                        if drop_mask is not None:
                            if isinstance(drop_mask, torch.Tensor):
                                valid_num = num_frames - 2 - torch.sum(drop_mask[0])

                                df = 2 + int(valid_num)
                                dk, dv = [], []
                                dk.append(k[:, :, :2])
                                dv.append(v[:, :, :2])

                                if valid_num != 0:
                                    ind = torch.arange(0, drop_mask.shape[1])[None].repeat(drop_mask.shape[0], 1).to(drop_mask.device)
                                    ind = ind[drop_mask == 0].reshape(drop_mask.shape[0], -1)

                                    bs, khw, n, kc = k.shape
                                    _, vhw, _, vc = v.shape
                                    ka = k[:, :, 2:].permute(0,1,3,2).reshape(bs, -1, n-2)
                                    va = v[:, :, 2:].permute(0,1,3,2).reshape(bs, -1, n-2)
                                    ind = ind[:,None].repeat(1, ka.shape[1], 1)
                                    ki = torch.gather(ka, 2, ind)
                                    vi = torch.gather(va, 2, ind)
                                    ki = ki.reshape(bs, khw, kc, -1).permute(0, 1, 3, 2)
                                    vi = vi.reshape(bs, vhw, vc, -1).permute(0, 1, 3, 2)

                                    dk.append(ki)
                                    dv.append(vi)

                                dk = torch.cat(dk, dim=2)
                                dv = torch.cat(dv, dim=2)
                        else:
                            dk, dv = k, v
                            df = num_frames

                        kt, vt = rearrange_many(
                            (dk, dv), "b s f (h c) -> (b h) (s f) c", h=h, f=df, s=s
                        )
                    else:
                        raise NotImplementedError
                elif attn_mode == 'add':
                    kt, vt = rearrange_many(
                        (k, v), "b s f (h c) -> (b h) s f c", h=h, f=num_frames, s=s
                    )
                    kt = torch.sum(kt, dim=2) / num_frames
                    vt = torch.sum(vt, dim=2) / num_frames

                    kt = kt.to(qt.dtype)
                    vt = vt.to(qt.dtype)
                elif attn_mode == 'random_mask':
                    kt, vt = rearrange_many(
                        (k, v), "b s f (h c) -> (b h s) f c", h=h, f=num_frames, s=s
                    )
                    bm, f, c = kt.shape
                    rand_mask = torch.randint(0, num_frames, (bm, 1, c)).to(kt.device)
                    mask = torch.zeros_like(kt)
                    mask.scatter_(1, rand_mask, torch.ones_like(rand_mask).to(kt))
                    kt = torch.sum(kt * mask, dim=1)
                    vt = torch.sum(vt * mask, dim=1)

                    kt, vt = rearrange_many(
                        (kt, vt), "(b h s) c -> (b h) s c", b= bs,  h=h, s=s
                    )

                    kt = kt.to(qt.dtype)
                    vt = vt.to(qt.dtype)
                elif attn_mode == 'sliding_window':
                    num = sliding_wondow.shape[1]
                    kt, vt = rearrange_many(
                        (k, v), "b s f (h c) -> (b h s) f c", h=h, f=num_frames, s=s
                    )
                    bm, f, c = kt.shape

                    sliding_ind = sliding_wondow.long().to(k.device)
                    sliding_ind = sliding_ind[:,None, :, None].repeat(1, h *s, 1, c)
                    sliding_ind = sliding_ind.reshape(bm, -1, c)

                    kt_m = torch.gather(kt, 1, sliding_ind)
                    vt_m = torch.gather(vt, 1, sliding_ind)

                    kt, vt = rearrange_many(
                        (kt_m, vt_m), "(b h s) f c -> (b h) (s f) c", h=h, f=num, s=s
                    )
                else:
                    raise NotImplementedError

                resq = xformers.ops.memory_efficient_attention(
                    qt, kt, vt, attn_bias=mask, op=self.attention_op
                )

                resq = rearrange(resq, "(b h) (s f) c -> b s f (h c)", h=h, f=1, s=s)

                if not self.anchor_infer_once or first:
                    res = torch.cat([resq, ress], dim=2)
                else:
                    res = resq

            else:
                q, k, v = rearrange_many(
                    (q, k, v), "b s f (h c) -> (b h) (s f) c", h=h, f=num_frames, s=s
                )  #  (bhw) heads, dim_head, 5

                if rotary_emb is not None:
                    q = rearrange(q, "b (s f) c -> b s f c", f=num_frames, s=s)
                    k = rearrange(k, "b (s f) c -> b s f c", f=num_frames, s=s)
                    q = rotary_emb.rotate_queries_or_keys(q)
                    k = rotary_emb.rotate_queries_or_keys(k)
                    q = rearrange(q, "b s f c -> b (s f) c", f=num_frames, s=s)
                    k = rearrange(k, "b s f c -> b (s f) c", f=num_frames, s=s)

                res = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )
                # res = F.scaled_dot_product_attention(
                #     q, k, v, dropout_p=0.0, is_causal=False
                # )

                res = rearrange(res, "(b h) (s f) c -> b s f (h c)", h=h, f=num_frames, s=s)

        return self.to_out(res)


class CrossAttentionTemporal3DV2(nn.Module):
    """Similar to CrossAttentionTemporal but fully attended"""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attend_prev_only=True,
        mask_mode="v",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attend_prev_only = attend_prev_only

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        low_rank = 8
        self.to_q_lora = nn.Sequential(
            nn.Linear(query_dim, low_rank, bias=False),
            zero_module(nn.Linear(low_rank, inner_dim, bias=False)),
        )
        self.to_k_lora = nn.Sequential(
            nn.Linear(context_dim, low_rank, bias=False),
            zero_module(nn.Linear(low_rank, inner_dim, bias=False)),
        )
        self.to_v_lora = nn.Sequential(
            nn.Linear(context_dim, low_rank, bias=False),
            zero_module(nn.Linear(low_rank, inner_dim, bias=False)),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            # nn.Dropout(dropout)
        )
        self.to_out_lora = nn.Sequential(
            nn.Linear(inner_dim, low_rank, bias=False),
            zero_module(nn.Linear(low_rank, query_dim)),
            # nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.attention_op: Optional[Any] = None
        self.mask_mode = mask_mode

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        pos_bias=None,
        focus_present_mask_only=False,
        t_temp=1.0,
        diag_offset=1,
    ):
        # b (h w) 5 c
        h = self.heads
        num_frames = x.shape[2]
        s = x.shape[1]
        bs = x.shape[0]
        device = x.device

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if focus_present_mask_only:
            if self.mask_mode == "2d":
                # Equivalent to MemoryEfficientCrossAttention of 2D Version
                q, k, v = rearrange_many(
                    (q, k, v), "b s f (h c) -> (b f h) s c", h=h, f=num_frames, s=s
                )  #  (bhw) heads, dim_head, 5

                res = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )

                res = rearrange(
                    res, "(b f h) s c -> b s f (h c)", h=h, f=num_frames, s=s
                )  #  (bhw) heads, dim_head, 5
            elif self.mask_mode == "v":
                res = v
            else:
                raise NotImplemented(f"unknown mask mode {mask_mode}")

            return self.dropout(self.to_out(res))
        else:
            q = q + self.to_q_lora(x)
            k = k + self.to_k_lora(context)
            v = v + self.to_v_lora(context)

            q, k, v = rearrange_many(
                (q, k, v), "b s f (h c) -> (b h) (s f) c", h=h, f=num_frames, s=s
            )  #  (bhw) heads, dim_head, 5

            if rotary_emb is not None:
                q = rearrange(q, "b (s f) c -> b s f c", f=num_frames, s=s)
                k = rearrange(k, "b (s f) c -> b s f c", f=num_frames, s=s)
                q = rotary_emb.rotate_queries_or_keys(q)
                k = rotary_emb.rotate_queries_or_keys(k)
                q = rearrange(q, "b s f c -> b (s f) c", f=num_frames, s=s)
                k = rearrange(k, "b s f c -> b (s f) c", f=num_frames, s=s)

            res = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

            res = rearrange(res, "(b h) (s f) c -> b s f (h c)", h=h, f=num_frames, s=s)

            return self.dropout(self.to_out(res) + self.to_out_lora(res))


ATTENTION_MODES = {
    "softmax": CrossAttention,  # vanilla attention
    "softmax-xformers": MemoryEfficientCrossAttention,
    "plain": CrossAttention_plain,
    "temporal": CrossAttentionTemporal_xformer,
    "temporal-3d": CrossAttentionTemporal3D,
    "temporal-3d-v2": CrossAttentionTemporal3DV2,
}


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in ATTENTION_MODES
        attn_cls = ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSpaceTimeBlock_Sequential(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attend_prev_only=True,
        use_rotary=False,
        diag_offset=1,
        use_pvt_sa=False,
        temporal_attn_mode="temporal",
        **kwargs,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        attn_cls = ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=None,
        )  # is a self-attention

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            **kwargs
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        temporal_attn_cls = ATTENTION_MODES[temporal_attn_mode]
        self.temporal_attn1 = temporal_attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            attend_prev_only=attend_prev_only,
        )
        self.temporal_norm1 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.diag_offset = diag_offset
        self.use_rotary = use_rotary
        if self.use_rotary:
            rot_dim = min(32, int(dim / n_heads))
            self.rotary_emb = RotaryEmbedding(rot_dim)
        else:
            self.rotary_emb = None

    def forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch="full",
    ):
        return checkpoint(
            self._forward,
            (x, context, focus_present_mask_only, pos_bias),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch=False,
    ):
        x = list(torch.split(x, 1, 2))  # [b,hw,1,c] * f
        for frame_i in range(len(x)):
            x[frame_i] = x[frame_i].squeeze(2)
            x[frame_i] = self.attn1(self.norm1(x[frame_i])) + x[frame_i]
            x[frame_i] = (
                 elf.attn2(self.norm2(x[frame_i]), context=context[:, frame_i])
                + x[frame_i]
            )
            x[frame_i] = self.ff(self.norm3(x[frame_i])) + x[frame_i]
            x[frame_i] = x[frame_i].unsqueeze(2)
        x = torch.cat(x, dim=2)

        # x: [b,hw,f,c]
        x = (
            self.temporal_attn1(
                self.temporal_norm1(x),
                context=None,
                rotary_emb=self.rotary_emb,
                pos_bias=pos_bias,
                focus_present_mask_only=focus_present_mask_only,
                t_temp=0.8,
                diag_offset=self.diag_offset,
            )
            + x
        )

        return x


class BasicTransformerSpaceTimeBlock_Sequential_V2(
    BasicTransformerSpaceTimeBlock_Sequential
):
    def _forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch=False,
    ):
        x = list(torch.split(x, 1, 2))  # [b,hw,1,c] * f
        for frame_i in range(len(x)):
            x[frame_i] = x[frame_i].squeeze(2)
            x[frame_i] = self.attn1(self.norm1(x[frame_i])) + x[frame_i]
            x[frame_i] = (
                self.attn2(self.norm2(x[frame_i]), context=context[:, frame_i])
                + x[frame_i]
            )
            x[frame_i] = self.ff(self.norm3(x[frame_i])) + x[frame_i]
            x[frame_i] = x[frame_i].unsqueeze(2)
        x = torch.cat(x, dim=2)

        # x: [b,hw,f,c]
        if not focus_present_mask_only:
            x = (
                self.temporal_attn1(
                    self.temporal_norm1(x),
                    context=None,
                    rotary_emb=self.rotary_emb,
                    pos_bias=pos_bias,
                    focus_present_mask_only=focus_present_mask_only,
                    t_temp=0.8,
                    diag_offset=self.diag_offset,
                )
                + x
            )

        return x


class BasicTransformerSpaceTimeBlock_Reshape(nn.Module):
    """Similar to BasicTransformerSpaceTimeBlock_Reshape but uses temporal-3d-v2 and therefore supports rotary"""

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attend_prev_only=True,
        use_rotary=False,
        diag_offset=1,
        use_pvt_sa=False,
        temporal_attn_mode="temporal-3d",
        use_joint_attn=False,
        use_multiextra=False,
        use_extra=False,
        use_temporal_attn=False,
        only_image=False,
        use_sep_selfattn=False,
        use_second_3d_attn=False,
        anchor_infer_once=False,
        second_3d_attn_mode='after',
        second_weights=1.,
        use_clean_second=False,
        train_4viewip=False,
        random_drop_rate=0.,
        **kwargs,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        attn_cls = ATTENTION_MODES[attn_mode]
        assert temporal_attn_mode in ["temporal-3d", "temporal-3d-v2", "plain"]
        temporal_attn_cls = ATTENTION_MODES[temporal_attn_mode]
        self.disable_self_attn = False
        self.attn1 = temporal_attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=None,
            mask_mode="2d",
            use_sep_selfattn=use_sep_selfattn,
            train_4viewip=train_4viewip,
            anchor_infer_once=anchor_infer_once,
        )  # is a self-attention

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            only_image=only_image,
            **kwargs
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.diag_offset = diag_offset
        self.use_rotary = use_rotary
        if self.use_rotary:
            rot_dim = min(32, int(dim / n_heads))
            self.rotary_emb = RotaryEmbedding(rot_dim)
        else:
            self.rotary_emb = None

        self.use_joint_attn = use_joint_attn
        self.use_multiextra = use_multiextra
        self.use_sep_selfattn = use_sep_selfattn
        self.train_4viewip = train_4viewip
        self.use_extra = use_extra
        self.anchor_infer_once = anchor_infer_once

        if use_joint_attn:
            temporal_attn_cls = ATTENTION_MODES["temporal-3d"]
            self.joint_attn = temporal_attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=None,
                mask_mode="2d",
            )
            self.norm4 = nn.LayerNorm(dim)

        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            temporal_attn_cls = ATTENTION_MODES["temporal"]
            self.temporal_attn = temporal_attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=None,
                mask_mode="2d",
            )
            self.norm5 = nn.LayerNorm(dim)

        self.use_second_3d_attn = use_second_3d_attn
        self.second_3d_attn_mode = second_3d_attn_mode
        self.second_weights = second_weights
        self.use_clean_second = use_clean_second
        if use_second_3d_attn:
            temporal_attn_cls = ATTENTION_MODES[temporal_attn_mode]
            self.second_3dattn = temporal_attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=None,
                mask_mode="2d",
                use_sep_selfattn=use_sep_selfattn,
                anchor_infer_once=anchor_infer_once,
                random_drop_rate=random_drop_rate
            )
            self.secondattn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch="full",
        selfattn_mode=3,
        attn_mode='org',
        slidewin=None,
        reset=False,
        second_weights=None,
        drop_mask=None,
    ):
        return checkpoint(
            self._forward,
            (x, context, focus_present_mask_only, pos_bias, selfattn_mode, attn_mode, slidewin, reset, second_weights, drop_mask),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        selfattn_mode=3,
        attn_mode='org',
        slidewin=None,
        reset=False,
        second_weights=None,
        drop_mask=None,
    ):
        """
            Idea is first spacetial temproal no context in the model
            in self attention of each frame we add context in
        """
        attn1_mode = selfattn_mode
        if self.train_4viewip or self.use_second_3d_attn:
            if self.use_clean_second:
                attn1_mode = 4
            else:
                attn1_mode = 1

        x = (
            self.attn1(
                self.norm1(x),
                context=None,
                rotary_emb=self.rotary_emb,
                pos_bias=pos_bias,
                focus_present_mask_only=focus_present_mask_only,
                t_temp=0.8,
                diag_offset=self.diag_offset,
                mode=attn1_mode,
                attn_mode=attn_mode,
                sliding_wondow=slidewin,
                reset=reset,
                drop_mask=drop_mask,
            )
            + x
        )
        bs, hw, f, n = x.shape

        x = rearrange(x, "b l f c -> (b f) l c", f=f).contiguous()
        context = rearrange(context, "b f l c -> (b f) l c", f=f).contiguous()
        x = (
            self.attn2(self.norm2(x), context=context) + x
        )
        x = rearrange(x , "(b f) l c -> b l f c", f=f).contiguous()

        x = self.ff(self.norm3(x)) + x

        if self.use_temporal_attn:
            x = (
                    self.temporal_attn(
                        self.norm5(x),
                        context=None,
                        rotary_emb=self.rotary_emb,
                        pos_bias=pos_bias,
                        focus_present_mask_only=focus_present_mask_only,
                        t_temp=0.8,
                        diag_offset=self.diag_offset,
                    )
                    + x
            )

        return x


class BasicTransformerSpaceTimeBlock_Reshape_V2(nn.Module):
    """Similar to BasicTransformerSpaceTimeBlock_Reshape but uses temporal-3d-v2 and therefore supports rotary"""

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attend_prev_only=True,
        use_rotary=False,
        diag_offset=1,
        use_pvt_sa=False,
        temporal_attn_mode="temporal-3d",
        emb_channels=1280,
        **kwargs,
    ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        attn_cls = ATTENTION_MODES[attn_mode]
        assert temporal_attn_mode in ["temporal-3d", "temporal-3d-v2"]
        temporal_attn_cls = ATTENTION_MODES[temporal_attn_mode]
        self.disable_self_attn = False
        self.pos_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 8, bias=False),
            zero_module(
                nn.Linear(
                    8,
                    dim,
                )
            ),
        )
        self.attn1 = temporal_attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=None,
            mask_mode="2d",
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            **kwargs
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        self.diag_offset = diag_offset

        self.use_rotary = use_rotary
        if self.use_rotary:
            rot_dim = min(32, int(dim / n_heads))
            self.rotary_emb = RotaryEmbedding(rot_dim)
        else:
            self.rotary_emb = None

    def forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch="full",
    ):
        return checkpoint(
            self._forward,
            (x, context, focus_present_mask_only, pos_bias),
            self.parameters(),
            self.checkpoint,
        )

    def temporal_params(self):
        return (
            list(self.pos_emb_layers.parameters())
            + list(self.norm1.parameters())
            + list(self.attn1.parameters())
        )

    def _forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch=False,
    ):
        if not focus_present_mask_only:
            pos_emb = self.pos_emb_layers(pos_bias)  # [b,f,c]
            x = x + pos_emb.unsqueeze(1)
        x = (
            self.attn1(
                self.norm1(x),
                context=None,
                rotary_emb=self.rotary_emb,
                pos_bias=pos_bias,
                focus_present_mask_only=focus_present_mask_only,
                t_temp=0.8,
                diag_offset=self.diag_offset,
            )
            + x
        )

        x = list(torch.split(x, 1, 2))  # [b,hw,1,c] * f
        for frame_i in range(len(x)):
            x[frame_i] = x[frame_i].squeeze(2)
            x[frame_i] = (
                self.attn2(self.norm2(x[frame_i]), context=context[:, frame_i])
                + x[frame_i]
            )
            x[frame_i] = self.ff(self.norm3(x[frame_i])) + x[frame_i]
            x[frame_i] = x[frame_i].unsqueeze(2)
        x = torch.cat(x, dim=2)
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialTemporalTransformer(nn.Module):
    """
        Transformer block for image-like data.
        First, project the input (aka embedding)
        and reshape to b, t, d.
        Then apply standard transformer action.
        Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attend_prev_only=True,
        use_rotary=False,
        out_channels=None,
        attention_block="sequential",
        diag_offset=1,
        use_pvt_sa=False,
        temporal_attn_mode="temporal",
        frame_num=8,
        use_scale_shift=False,
        emb_channels=1280,
        use_joint_attn=False,
        use_temporal_attn=False,
        only_image=False,
        use_extra=False,
        use_multiextra=False,
        use_sep_selfattn=False,
        use_second_3d_attn=False,
        second_3d_attn_mode='after',
        anchor_infer_once=False,
        second_weights=1.,
        use_clean_second=False,
        train_4viewip=False,
        random_drop_rate=0.,
        **kwargs
    ):
        super().__init__()
        print(
            f"[SpatialTemporalTransformer] => attention_block: {attention_block} attend_prev_only: {attend_prev_only} use_rotary: {use_rotary} temporal_attn_mode: {temporal_attn_mode}"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.window_t_attn = False
        if attention_block == "sequential":
            attention_block = BasicTransformerSpaceTimeBlock_Sequential
        elif attention_block == "sequential_v2":
            attention_block = BasicTransformerSpaceTimeBlock_Sequential_V2
        elif attention_block == "reshape":
            attention_block = BasicTransformerSpaceTimeBlock_Reshape
        elif attention_block == "reshape_v2":
            attention_block = BasicTransformerSpaceTimeBlock_Reshape_V2
        else:
            raise ValueError(f"Unknown attention block type: {attention_block}")

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.use_scale_shift = use_scale_shift
        if use_scale_shift:
            self.input_scale, self.input_shift = nn.Parameter(
                torch.ones((inner_dim, frame_num))
            ), nn.Parameter(torch.zeros((inner_dim, frame_num)))

        self.transformer_blocks = nn.ModuleList(
            [
                attention_block(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attend_prev_only=attend_prev_only,
                    use_rotary=use_rotary,
                    diag_offset=diag_offset,
                    use_pvt_sa=use_pvt_sa,
                    temporal_attn_mode=temporal_attn_mode,
                    emb_channels=emb_channels,
                    use_joint_attn=use_joint_attn,
                    use_temporal_attn=use_temporal_attn,
                    only_image=only_image,
                    use_extra=use_extra,
                    use_multiextra=use_multiextra,
                    use_sep_selfattn=use_sep_selfattn,
                    use_second_3d_attn=use_second_3d_attn,
                    second_3d_attn_mode=second_3d_attn_mode,
                    second_weights=second_weights,
                    use_clean_second=use_clean_second,
                    train_4viewip=train_4viewip,
                    anchor_infer_once=anchor_infer_once,
                    random_drop_rate=random_drop_rate,
                    **kwargs
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

        self.use_linear = use_linear

    def forward(
        self,
        x,
        context=None,
        focus_present_mask_only=False,
        pos_bias=None,
        return_single_branch="full",
        selfattn_mode=3,
        attn_mode='org',
        slidewin=None,
        reset=False,
        second_weights=None,
        drop_mask=None,
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, f, h, w = x.shape
        x_in = x

        x = rearrange(x, "b c f h w -> (b f) c h w").contiguous()
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "(b f) c h w -> (b f) (h w) c", f=f).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "(b f) (h w) c -> b (h w) f c", f=f, h=h, w=w).contiguous()

        if self.use_scale_shift and not focus_present_mask_only:
            x = self.input_scale.permute(1, 0) * x + self.input_shift.permute(1, 0)

        for block in self.transformer_blocks:
            if self.window_t_attn:
                for idx in [0, 10]:
                    x[:, :, idx : idx + 10] = block(
                        x[:, :, idx : idx + 10],
                        context=context,
                        focus_present_mask_only=focus_present_mask_only,
                        pos_bias=pos_bias[:, idx : idx + 10, idx : idx + 10]
                        if pos_bias is not None
                        else pos_bias,
                        return_single_branch=return_single_branch,
                    )
            else:
                x = block(
                    x,
                    context=context,
                    focus_present_mask_only=focus_present_mask_only,
                    pos_bias=pos_bias,
                    return_single_branch=return_single_branch,
                    selfattn_mode=selfattn_mode,
                    attn_mode=attn_mode,
                    slidewin=slidewin,
                    reset=reset,
                    second_weights=second_weights,
                    drop_mask=drop_mask,
                )

        x = rearrange(x, "b (h w) f c -> (b f) (h w) c", h=h, w=w).contiguous()
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "(b f) (h w) c -> (b f) c h w", h=h, w=w, f=f).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", h=h, w=w, f=f).contiguous()

        return x + x_in  # if x.shape == x_in.shape else x
