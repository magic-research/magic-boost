from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many, check_shape

from mvc.ldm.modules.diffusionmodules.resampler import (
    Resampler, 
    SimpleReSampler
)
from mvc.ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from mvc.ldm.modules.attention import SpatialTransformer, SpatialTemporalTransformer, exists
import pdb
b=pdb.set_trace


class LayerNorm(nn.Module):
    def __init__(self, dim, stable = False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(th.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        eps = 1e-5 if x.dtype == th.float32 else 1e-3
        var = th.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = th.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        depth
    ):
        super().__init__()
        self.temporal_mlp = nn.ModuleList([])
        self.temporal_mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim),
            nn.SiLU()
        ))

        for _ in range(max(depth - 1, 0)):
            self.temporal_mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                LayerNorm(dim),
                nn.SiLU()
            ))

        self.temporal_mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):
        i = th.arange(n, device = device)
        j = th.arange(n, device = device)

        indices = rearrange(i, 'i -> i 1') - rearrange(j, 'j -> 1 j')
        indices += (n - 1)

        pos = th.arange(-n + 1, n, device = device, dtype = dtype)
        pos = rearrange(pos, '... -> ... 1')

        for layer in self.temporal_mlp:
            pos = layer(pos)

        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

# dummy replace
def convert_module_to_f16(l):
    """
        Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
        A sequential module that passes timestep embeddings to the children that
        support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class TimestepEmbedSequentialSpaceTime(nn.Sequential, TimestepBlock):
    """
        A sequential module that passes timestep embeddings to the children that
        support it as an extra input.
    """

    def forward(self, x, emb, 
                context=None, 
                pos_bias=None, 
                focus_present_mask_only=False, 
                return_single_branch='full',
                selfattn_mode=3,
                attn_mode='org',
                slidewin=None,
                reset=False,
                second_weights=None,
                drop_mask=None):
        # check_shape(x, 'b c f h w', f = 5)
        for layer in self:
            if isinstance(layer, SpatialTemporalTransformer):
                x = layer(x, context,
                        focus_present_mask_only=focus_present_mask_only,
                        pos_bias=pos_bias,
                        return_single_branch=return_single_branch,
                        selfattn_mode=selfattn_mode,
                        attn_mode=attn_mode,
                        slidewin=slidewin,
                        reset=reset,
                        second_weights=second_weights,
                        drop_mask=drop_mask)
            elif isinstance(layer, TemporalResBlock):
                x = layer(x, emb, focus_present_mask_only=focus_present_mask_only)
            elif isinstance(layer, PseudoConv3d):
                x = layer(x, focus_present_mask_only=focus_present_mask_only)
            elif isinstance(layer, TimestepBlock):
                x = list(th.split(x, 1, 2)) if not isinstance(x, list) else x 
                frames_n = len(x)
                for frame_i in range(frames_n):
                    x[frame_i] = layer(x[frame_i].squeeze(2), emb[:,frame_i]).unsqueeze(2)
                x = th.cat(x, 2)
            elif isinstance(layer, SpatialTransformer):
                x = list(th.split(x, 1, 2)) if not isinstance(x, list) else x 
                frames_n = len(x)
                for frame_i in range(frames_n):
                    x[frame_i] = layer(x[frame_i].squeeze(2), context).unsqueeze(2)
                x = th.cat(x, 2)
            else:
                x = list(th.split(x, 1, 2)) if not isinstance(x, list) else x 
                frames_n = len(x)
                for frame_i in range(frames_n):
                    x[frame_i] = layer(x[frame_i].squeeze(2)).unsqueeze(2)
                x = th.cat(x, 2)        
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
        A residual block that can optionally change the number of channels.
        :param channels: the number of input channels.
        :param emb_channels: the number of timestep embedding channels.
        :param dropout: the rate of dropout.
        :param out_channels: if specified, the number of out channels.
        :param use_conv: if True and out_channels is specified, use a spatial
            convolution instead of a smaller 1x1 convolution to change the
            channels in the skip connection.
        :param dims: determines if the signal is 1D, 2D, or 3D.
        :param use_checkpoint: if True, use gradient checkpointing on this module.
        :param up: if True, use this block for upsampling.
        :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out if emb is not None else h
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class PseudoConv3d(nn.Module):
    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size,
        padding = 0,
        identity_init = False,
    ):
        super().__init__()
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        if ndim == 1:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
            self.padding = (padding, 0, 0,)
        elif ndim == 2:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.padding = (0, padding, padding)
        else:
            raise NotImplementedError(f"ndim={ndim} not supported")
        self.bias = nn.Parameter(torch.zeros(out_channels))

        if identity_init:
            nn.init.dirac_(self.weight.data) # initialized to be identity
        else:
            nn.init.kaiming_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def forward(self, x, focus_present_mask_only=False):
        if self.ndim == 1:
            if focus_present_mask_only:
                # skip temporal convolution
                return x
            weight = self.weight.unsqueeze(3).unsqueeze(4)
        elif self.ndim == 2:
            weight = self.weight.unsqueeze(2)
        x = F.conv3d(x, weight=weight, bias=self.bias, padding=self.padding)
        return x

class TemporalResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2, # not used
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            PseudoConv3d(2, channels, self.out_channels, 3, padding=1),
        )
        self.in_conv_temp = PseudoConv3d(1, self.out_channels, self.out_channels, 3, padding=1, identity_init=True)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, 3)
            self.x_upd = Upsample(channels, False, 3)
        elif down:
            self.h_upd = Downsample(channels, False, 3)
            self.x_upd = Downsample(channels, False, 3)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                PseudoConv3d(2, self.out_channels, self.out_channels, 3, padding=1),
            ),
        )
        self.out_conv_temp = PseudoConv3d(1, self.out_channels, self.out_channels, 3, padding=1, identity_init=True)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = PseudoConv3d(2, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = PseudoConv3d(2, channels, self.out_channels, 1)

    def convert_3d_to_2d(self, x, shape):
        N, C, F, H, W = shape
        x = x.permute(0,2,1,3,4).reshape(N*F,C,H,W).contiguous()
        return x

    def convert_2d_to_3d(self, x, shape):
        N, C, F, H, W = shape
        x = x.reshape(N, F, C, H, W).permute(0,2,1,3,4).contiguous()
        return x

    def convert_2d_to_1d(self, x, shape):
        N, C, F, H, W = shape
        x = x.reshape(N, F, C, H, W).permute(0,3,4,2,1).reshape(N*H*W,C,F).contiguous()
        return x

    def convert_1d_to_2d(self, x, shape):
        N, C, F, H, W = shape
        x = x.reshape(N, H, W, C, F).permute(0,4,3,1,2).reshape(N*F,C,H,W).contiguous()
        return x

    def forward(self, x, emb, focus_present_mask_only=False):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, focus_present_mask_only), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, focus_present_mask_only=False):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if not focus_present_mask_only:
            h = self.in_conv_temp(h)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            if emb_out.ndim == 3:
                emb_out = emb_out.permute(0,2,1)
            else:
                assert emb_out.ndim == 2
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out if emb is not None else h
            h = self.out_layers(h)
        if not focus_present_mask_only:
            h = self.out_conv_temp(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class FusedMultiViewUNetModel(UNetModel):
    def __init__(self, *args, frame_num=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = frame_num
        self._forced_present_mask = False

    def set_forced_present_mask(self, value):
        assert isinstance(value, bool)
        self._forced_present_mask = value

    def forward(self, x, timesteps=None, context=None, focus_present_mask_only=False, fps=None):
        if focus_present_mask_only or self._forced_present_mask:
            return super().forward(x, timesteps, context=context)
        else:
            B, C, H, W = x.shape
            assert math.sqrt(self.frame_num) % 1 == 0
            assert B % self.frame_num == 0
            nH = nW = int(math.sqrt(self.frame_num))
            x = rearrange(x, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nH, nw=nW)
            timesteps = timesteps.reshape(-1, self.frame_num)[:,0]
            context = context.reshape(-1, self.frame_num, *context.shape[1:])[:,0]
            h = super().forward(x, timesteps, context=context)
            h = rearrange(h, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nH, nw=nW)
            return h


class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, 
            cross_attention_dim=1024, 
            clip_embeddings_dim=1024, 
            clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        # from 1024 -> 4 * 1024
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, 
            self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class UNetModelSpaceTime(nn.Module):
    """
    [Modified to adapt to videogen stv3 structure]
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size, # not used
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks, # 2 Official 
        attention_resolutions, # [4, 2, 1] Official
        dropout=0,
        channel_mult=(1, 2, 4, 8), # [1, 2, 4, 4] Official 
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,  # 8 Official 
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support: Official True 
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support: Official 768 
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_linear_in_transformer=False,
        use_rotary=False, 
        attend_prev_only=False, 
        no_pos_bias=True, 
        temporal_attention_resolutions=None,
        use_scale_adjustment=False,
        fps_add_to_emb=False,
        frame_num=4,
        attention_block='reshape',
        diag_offset=1,
        use_pvt_sa=False,
        temporal_conv=False,
        temporal_attn_mode="temporal-3d",
        attention_scale_shift=False,
        frame_add_to_emb=False,
        camera_dim=16,
        camera_emb_mode="time_emb",
        camera_depth=2,
        image_cond_channels=0,
        image_cond_mode="input",
        image_cond_aug=False,
        with_ip=False,  # wether add image prompt images
        ip_num=1, # number of prompt images
        ip_dim=0, # number of extra token, 4 for global 16 for local
        ip_weight=1.0, # weight for image prompt context
        ip_mode="",
        use_joint_attn=False,
        use_joint_pos='mid',
        use_temporal_attn=False,
        use_temporal_pos='all',
        use_second_3d_attn=False,
        sep_depth_branch=False,
        only_image=False,
        use_extra=False,
        use_multiextra=False,
        use_sep_selfattn=False,
        train_4viewip=False,
        use_onetime_anchor=False,
        second_3d_attn_mode='after',
        second_weights=1.,
        anchor_infer_once=False,
        use_clean_second=False,
        random_drop_rate=0.,
        use_split_noise_embedding=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        # self.use_scale_adjustment = use_scale_adjustment
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        # import pdb; pdb.set_trace()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.temporal_attention_resolutions = temporal_attention_resolutions or attention_resolutions
        self.fps_add_to_emb=fps_add_to_emb
        self.frame_num = frame_num
        self.temporal_conv = temporal_conv
        self.frame_add_to_emb = frame_add_to_emb
        self.camera_dim = camera_dim
        self.camera_emb_mode = camera_emb_mode
        self.image_cond_channels = image_cond_channels
        self.image_cond_mode = image_cond_mode
        self.image_cond_aug = image_cond_aug
        self.with_ip = with_ip # wether there is image prompt
        self.ip_num = ip_num # num of ip images
        self.ip_dim = ip_dim # num of context dimension
        self.ip_weight = ip_weight
        self.ip_mode = ip_mode # which mode of adaptor
        self._forced_present_mask = False
        self._force_mode_type = None
        self._force_attn_mode_type = None
        self._force_slidingwin = None
        self._force_reset = False
        self._force_second_weigths = None
        self._force_drop_mask = None
        self.use_joint_attn = use_joint_attn
        self.use_temporal_attn = use_temporal_attn
        self.only_image = only_image
        self.use_extra = use_extra
        self.use_multiextra = use_multiextra
        self.use_sep_selfattn = use_sep_selfattn
        self.use_second_3d_attn = use_second_3d_attn
        self.second_3d_attn_mode = second_3d_attn_mode
        self.second_weights = second_weights
        self.train_4viewip = train_4viewip
        self.use_onetime_anchor = use_onetime_anchor
        self.anchor_infer_once = anchor_infer_once
        self.use_clean_second = use_clean_second
        self.random_drop_rate = random_drop_rate
        self.use_split_noise_embedding = use_split_noise_embedding
        self.drop_mask = None

        if self.anchor_infer_once:
            self.counter = 0

        res_block_cls = TemporalResBlock if temporal_conv else ResBlock
        print(f"ResNet Block: [{res_block_cls}]")

        self.use_pvt_sa = use_pvt_sa
        self.attn_blk = attention_block
        self.no_pos_bias = no_pos_bias

        if not self.no_pos_bias:
            attn_heads = num_heads
            ch = int(channel_mult[0] * model_channels)
            self.time_rel_pos_bias = DynamicPositionBias(dim = ch * 2, heads = attn_heads, depth = 2)
            print("[UNetModelSpaceTimeV2] => Using DynamicPositionBias: dim:{} heads:{} depth:{}. ".format(ch * 2, attn_heads, 2))

        else:
            print("[UNetModelSpaceTimeV2] => Not using time_rel_pos_bias! ")

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.image_cond_aug:
            self.time_embed_aug = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.fps_add_to_emb:
            self.temporal_fps_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.frame_add_to_emb:
            self.frame_params = nn.Parameter(torch.ones((frame_num, time_embed_dim)))
            self.frame_embed = nn.Sequential(
                linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.use_split_noise_embedding:
            self.noise_embed = nn.Sequential(
                linear(2, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.camera_dim > 0:
            assert self.camera_emb_mode in ["context", "time_emb", "attn_emb"]
            camera_embed = []
            in_dim = self.camera_dim
            out_dim = 4 * context_dim if self.camera_emb_mode == "context" else time_embed_dim
            for _ in range(camera_depth-1):
                camera_embed.append( linear(in_dim, out_dim) )
                camera_embed.append( nn.SiLU() )
                in_dim = out_dim 

            camera_embed.append( linear(in_dim, out_dim) )
            self.camera_embed = nn.Sequential(*camera_embed)

        if self.with_ip and (context_dim is not None) and ip_dim > 0:
            if self.ip_mode == "":
                self.image_embed = ImageProjModel(
                    cross_attention_dim=context_dim, 
                    clip_extra_context_tokens=ip_dim)

            elif self.ip_mode == "plus":
                #ip-adapter-plus
                hidden_dim = 1280
                self.image_embed = Resampler(
                    dim=context_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=ip_dim, # num token
                    embedding_dim=hidden_dim,
                    output_dim=context_dim,
                    ff_mult=4
                )
            elif self.ip_mode == "local":
                self.image_embed = SimpleReSampler(
                    embedding_dim=1280, 
                    output_dim=context_dim
                )
            else:
                raise ValueError("not supported ip_mode")

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        if image_cond_channels > 0:
            if image_cond_mode in ["input", "upsample"]:
                self.input_cond_image = conv_nd(dims, 
                    image_cond_channels, model_channels, 3, padding=1)
            elif image_cond_mode == "context":
                self.input_cond_image = nn.Sequential(
                        conv_nd(dims, image_cond_channels, context_dim//4, 3, padding=1),
                        nn.SiLU(),
                        conv_nd(dims, context_dim//4, context_dim//2, 3, padding=1, stride=2),
                        nn.SiLU(),
                        conv_nd(dims, context_dim//2, context_dim, 3, padding=1, stride=2),
                )
                
            else:
                raise NotImplementedError(f"Unknown image_cond_mode={image_cond_mode}")

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequentialSpaceTime(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.sep_depth_branch = sep_depth_branch
        # if self.sep_depth_branch:
        #     self.sep_input_blocks0 = TimestepEmbedSequentialSpaceTime(
        #             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        #         )

        if temporal_conv:
            self.input_blocks[0].append(
                PseudoConv3d(1, 
                             model_channels, 
                             model_channels, 
                             3, 1, 
                             identity_init=True))

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    res_block_cls(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if ds == 1 and self.use_pvt_sa:
                        use_pvt_sa = self.use_pvt_sa
                    else:
                        use_pvt_sa=False
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if ds in self.temporal_attention_resolutions:
                        layers.append(
                            SpatialTemporalTransformer(
                                ch, num_heads, dim_head, 
                                depth=transformer_depth, 
                                context_dim=context_dim,  
                                disable_self_attn=False, 
                                use_linear=use_linear_in_transformer,
                                attend_prev_only=attend_prev_only, 
                                use_rotary=use_rotary, 
                                attention_block=self.attn_blk, 
                                diag_offset=diag_offset, 
                                use_pvt_sa = use_pvt_sa, 
                                temporal_attn_mode=temporal_attn_mode,
                                frame_num=frame_num, 
                                use_scale_shift=attention_scale_shift, 
                                emb_channels=time_embed_dim,
                                with_ip=self.with_ip,
                                ip_dim=self.ip_dim, 
                                ip_weight=self.ip_weight,
                                only_image=self.only_image,
                                use_joint_attn=self.use_joint_attn if use_joint_pos == 'down' or use_joint_pos == 'all' else False,
                                use_temporal_attn=self.use_temporal_attn if use_temporal_pos == 'down' or use_temporal_pos == 'all' else False,
                                use_second_3d_attn=self.use_second_3d_attn,
                                anchor_infer_once=self.anchor_infer_once,
                                second_3d_attn_mode=self.second_3d_attn_mode,
                                second_weights=self.second_weights,
                                use_clean_second=use_clean_second,
                                use_extra=self.use_extra,
                                use_multiextra=self.use_multiextra,
                                use_sep_selfattn=self.use_sep_selfattn,
                                train_4viewip=self.train_4viewip,
                                random_drop_rate=self.random_drop_rate,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, 
                                depth=transformer_depth, 
                                context_dim=context_dim
                            )
                        )

                self.input_blocks.append(TimestepEmbedSequentialSpaceTime(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialSpaceTime(
                        res_block_cls(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequentialSpaceTime(
            res_block_cls(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTemporalTransformer(
                    ch, num_heads, dim_head, 
                    depth=transformer_depth, 
                    context_dim=context_dim, 
                    disable_self_attn=False, 
                    use_linear=use_linear_in_transformer,
                    attend_prev_only=attend_prev_only, 
                    use_rotary=use_rotary, 
                    attention_block=self.attn_blk, 
                    diag_offset=diag_offset, 
                    temporal_attn_mode=temporal_attn_mode,
                    frame_num=frame_num, 
                    use_scale_shift=attention_scale_shift, 
                    emb_channels=time_embed_dim,
                    with_ip=self.with_ip,
                    ip_dim=self.ip_dim, 
                    ip_weight=self.ip_weight,
                    use_joint_attn=self.use_joint_attn if use_joint_pos == 'mid' or use_joint_pos == 'all' else False,
                    use_temporal_attn=self.use_temporal_attn if use_temporal_pos == 'mid' or use_temporal_pos == 'all' else False,
                    use_second_3d_attn=self.use_second_3d_attn,
                    anchor_infer_once=self.anchor_infer_once,
                    second_3d_attn_mode=self.second_3d_attn_mode,
                    second_weights=self.second_weights,
                    use_clean_second=use_clean_second,
                    only_image=self.only_image,
                    use_extra=self.use_extra,
                    use_multiextra=self.use_multiextra,
                    use_sep_selfattn=self.use_sep_selfattn,
                    train_4viewip=self.train_4viewip,
                    random_drop_rate=self.random_drop_rate,
            ),
            res_block_cls(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    res_block_cls(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, 
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if ds == 1 and self.use_pvt_sa:
                        use_pvt_sa = self.use_pvt_sa
                    else:
                        use_pvt_sa=False

                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                        
                    if ds in self.temporal_attention_resolutions:
                        layers.append(
                            SpatialTemporalTransformer(
                                ch, num_heads, dim_head, 
                                depth=transformer_depth, 
                                context_dim=context_dim, 
                                disable_self_attn=False, 
                                use_linear=use_linear_in_transformer,
                                attend_prev_only=attend_prev_only, 
                                use_rotary=use_rotary, 
                                attention_block=self.attn_blk, 
                                diag_offset=diag_offset, 
                                use_pvt_sa = use_pvt_sa, 
                                temporal_attn_mode=temporal_attn_mode,
                                frame_num=frame_num, 
                                use_scale_shift=attention_scale_shift, 
                                emb_channels=time_embed_dim,
                                with_ip=self.with_ip,
                                ip_dim=self.ip_dim, 
                                ip_weight=self.ip_weight,
                                only_image=self.only_image,
                                use_temporal_attn=self.use_temporal_attn if use_temporal_pos == 'up' or use_temporal_pos == 'all' else False,
                                use_joint_attn=self.use_joint_attn if use_joint_pos == 'up' or use_joint_pos == 'all' else False,
                                use_second_3d_attn=self.use_second_3d_attn,
                                anchor_infer_once=self.anchor_infer_once,
                                second_3d_attn_mode=self.second_3d_attn_mode,
                                second_weights=self.second_weights,
                                use_clean_second=use_clean_second,
                                use_extra=self.use_extra,
                                use_multiextra=self.use_multiextra,
                                use_sep_selfattn=self.use_sep_selfattn,
                                train_4viewip=self.train_4viewip,
                                random_drop_rate=self.random_drop_rate,

                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        res_block_cls(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequentialSpaceTime(*layers))
                self._feature_size += ch

        self.out = TimestepEmbedSequentialSpaceTime(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),   
        )
        if temporal_conv:
            self.out.append(PseudoConv3d(1, out_channels, out_channels, 3, 1, identity_init=True))

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

        self.use_scale_adjustment = use_scale_adjustment
        if self.use_scale_adjustment:
            # add scale and shift

            # set input block parameters
            self.input_scale=torch.nn.ParameterList([])
            self.input_shift=torch.nn.ParameterList([])
            for idx, module in enumerate(self.input_blocks):
                feature_num = module[-1].out_channels
                i_scale, i_shift = nn.Parameter(
                    torch.ones((feature_num, frame_num))), \
                    nn.Parameter(torch.zeros((feature_num, frame_num)))

                # import pdb; pdb.set_trace()
                self.input_scale.append(i_scale)
                self.input_shift.append(i_shift)

            # set middle block parameters
            feature_num = self.middle_block[-1].out_channels
            self.mid_scale, self.mid_shift = nn.Parameter(torch.ones((feature_num, frame_num))), \
                nn.Parameter(torch.zeros((feature_num, frame_num)))

            # set output block parameters
            self.output_scale=torch.nn.ParameterList([])
            self.output_shift=torch.nn.ParameterList([])
            for idx, module in enumerate(self.output_blocks):
                feature_num = module[-1].out_channels
                o_scale, o_shift = nn.Parameter(torch.ones((feature_num, frame_num))), \
                    nn.Parameter(torch.zeros((feature_num, frame_num)))

                # import pdb; pdb.set_trace()
                self.output_scale.append(o_scale)
                self.output_shift.append(o_shift)


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def set_forced_present_mask(self, value):
        assert isinstance(value, bool)
        self._forced_present_mask = value

    def set_forced_reset(self, value):
        assert isinstance(value, bool)
        self._force_reset = value

    def set_forced_mode_type(self, value):
        self._force_mode_type = value

    def set_forced_attn_mode_type(self, value):
        self._force_attn_mode_type = value

    def set_forced_slidingwin(self, value):
        self._force_slidingwin = value

    def set_second_weights(self, value):
        self._force_second_weigths = value

    def set_drop_mask(self, value):
        self._force_drop_mask = value

    def forward(self, x, timesteps=None,
                context=None,
                focus_present_mask_only=False,
                selfattn_mode=3,
                attn_mode='org',
                slidewin=None,
                reset=False,
                second_weights=None,
                drop_mask=None,
                fps=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param fps: fps value; currently not used. 
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param focus_present_mask_only: whether to use image only mode for training
        :return: an [N x C x ...] Tensor of outputs.
        """
        N, C, H, W = x.size()
        if self._forced_present_mask:
            focus_present_mask_only = True # override

        if self._force_mode_type is not None:
            selfattn_mode = self._force_mode_type

        if self._force_attn_mode_type is not None:
            attn_mode = self._force_attn_mode_type

        if self._force_slidingwin is not None:
            slidewin = self._force_slidingwin

        if self._force_reset:
            reset = self._force_reset
            self._force_reset = False
            self.counter = 0

        if self._force_second_weigths is not None:
            second_weights = self._force_second_weigths

        if self._force_drop_mask is not None:
            drop_mask = self._force_drop_mask

        add_cam = False
        if "add_cam" in context.keys():
            add_cam = context["add_cam"]
            del context["add_cam"]

        # Reshape 1: image and condition
        assert N % self.frame_num == 0, f"batch number {N} and {self.frame_num}"
        B = N // self.frame_num
        # (b f c h w) -> (b c f h w)
        x = x.reshape(-1, self.frame_num, C, H, W).permute(0,2,1,3,4) 
        
        # Get conditions        
        h_cond = None
        emb_aug = None
        if (self.camera_dim > 0 and not focus_present_mask_only) or add_cam:
            assert isinstance(context, dict)
            camera = context["camera"]

        if self.ip_dim > 0:
            assert isinstance(context, dict)
            ip = context["ip"]
            # fc = ip.shape[1] # frame num
            # ip = rearrange(ip, "b fc c d -> (b fc) c d")

        if "ip_img" in context:
            # for training we directly replace image in x
            # do transform encoding of ip image & 
            # N = batch size * 5
            # assert x.shape[2] == 5 and self.frame_num % 2 == 1
            x[:, :, 4, :, :] = context["ip_img"]

        if "ext_ip_img" in context:
            # assert x.shape[2] == 8
            exp_ip = context["ext_ip_img"].permute(0,2,1,3,4)
            ec = exp_ip.shape[2]
            x[:, :, -ec:, :, :] = exp_ip

            if self.use_onetime_anchor:
                timesteps = timesteps.reshape(-1, self.frame_num)
                timesteps[:, -ec:] = 0
                timesteps = timesteps.reshape(-1)

        if "view0" in context:
            x[:, :4, 0, :, :] = context["view0"]

        noise_embed = None
        if 'noise_embed' in context.keys():
            noise_embed = context['noise_embed']

        if self.image_cond_channels > 0:
            assert isinstance(context, dict)
            if "cond_image" in context:
                assert self.image_cond_channels > 0
                cond_image = context["cond_image"]

                if self.image_cond_mode == "input":
                    cond_image = rearrange(cond_image, "b fc c h w -> b (fc c) h w")
                    h_cond = self.input_cond_image(cond_image).unsqueeze(2) # b c 1 h w

                elif self.image_cond_mode == "upsample":
                    assert cond_image.shape[:3] == x.shape[:3]
                    fc = cond_image.shape[1]
                    cond_image = rearrange(cond_image, "b fc c h w -> (b fc) c h w")
                    h_cond = self.input_cond_image(cond_image)
                    h_cond = rearrange(h_cond, "(b fc) c h w -> b c fc h w", fc=fc)

                elif self.image_cond_mode == "context":
                    fc = cond_image.shape[1]
                    cond_image = rearrange(cond_image, "b fc c h w -> (b fc) c h w")
                    h_cond = self.input_cond_image(cond_image)
                    context = rearrange(h_cond, "(b fc) c h w -> b (fc h w) c", fc=fc).repeat_interleave(self.frame_num, dim=0)
                    h_cond = None

                else:
                    raise NotImplementedError(f"Unknown image_cond_mode={self.image_cond_mode}")

                if self.image_cond_aug:
                    t_aug = context.get("t_aug", torch.zeros(B).to(timesteps.device).type(torch.long))
                    t_emb_aug = timestep_embedding(t_aug, self.model_channels, repeat_only=False)
                    emb_aug = self.time_embed_aug(t_emb_aug).type(self.dtype)
                    emb_aug = emb_aug.repeat_interleave(self.frame_num, dim=0)

        if context["context"] is None:
            context = ip
        elif not isinstance(context, torch.Tensor):
            assert isinstance(context, dict)
            context = context["context"]
        context = context.reshape(-1, self.frame_num, *context.shape[1:])

        frames_n = x.shape[2]
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb).type(self.dtype)
        emb_original = emb.clone()

        if emb_aug is not None:
            emb = emb + emb_aug

        if focus_present_mask_only:
            fps_emb = None 
        else:
            if self.fps_add_to_emb:
                # import pdb; pdb.set_trace()
                if fps is None:
                    fps = torch.ones_like(timesteps) * 20
                fps_emb = self.temporal_fps_embed(
                    timestep_embedding(fps, self.model_channels, 20).type(x.dtype)) # (bs, 512x4)
            else:
                fps_emb = None
        
        # import pdb; pdb.set_trace()
        if fps_emb is not None:
            emb = emb + fps_emb
        
        # Reshape 2:  time embedding as well 
        emb = emb.reshape(-1, self.frame_num, emb.size(1))
        if self.frame_add_to_emb and not focus_present_mask_only:
            frame_emb = self.frame_embed(self.frame_params)
            emb = emb + frame_emb.unsqueeze(0)

        if self.use_split_noise_embedding and noise_embed is not None:
            noise_emb = self.noise_embed(noise_embed)
            emb = emb + noise_emb.reshape(-1, self.frame_num, noise_emb.size(1))
        else:
            camera[:, 2:4] += noise_embed

        if self.no_pos_bias or focus_present_mask_only:
            time_rel_pos_bias = None
        else:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device, dtype=x.dtype)

        if (self.camera_dim > 0 and not focus_present_mask_only and camera is not None) or add_cam:
            camera_emb = self.camera_embed(camera)
            camera_emb = camera_emb.reshape(-1, self.frame_num, camera_emb.shape[1])
            if self.camera_emb_mode == "context":
                # 4 extra token length
                camera_emb = camera_emb.unsqueeze(2).reshape(camera_emb.shape[0],camera_emb.shape[1],4,-1)
                context = torch.cat((context, camera_emb), 2) # what is this shape [bz, frame_num, token_len, dim]
            elif self.camera_emb_mode == "time_emb":
                emb = emb + camera_emb
            elif self.camera_emb_mode == "attn_emb":
                time_rel_pos_bias = camera_emb # override!
            else:
                raise NotImplementedError(f"Unknown camera_emb_mode={self.camera_emb_mode}")

        if self.ip_dim > 0 and not self.only_image:
            ip_emb = self.image_embed(ip)
            ip_emb = ip_emb.reshape(-1, self.frame_num, *ip_emb.shape[1:])
            self.anchor_ip_emb = ip_emb
            context = torch.cat((context, ip_emb), 2)
        else:
            context = context[:, :, None, :]
            ip_emb = None

        h = x.type(self.dtype)

        context = context.type(self.dtype)

        if self.anchor_infer_once and self.counter > 0:
            h = h[:, :, :1]
            emb = emb[:, :1]
            context = context[:, :1]

        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context, time_rel_pos_bias, focus_present_mask_only, selfattn_mode=selfattn_mode, attn_mode=attn_mode, slidewin=slidewin, reset=reset, second_weights=second_weights, drop_mask=drop_mask)

            if idx == 0 and self.sep_depth_branch:
                h = h / 2.

            if h_cond is not None:
                if h.shape[-2:] != h_cond.shape[-2:]:
                    raise ValueError(
                        f"Shape mismatch between input ({h.shape}) \
                          and cond_image ({h_cond.shape}). \
                          Resize images in the input domain!")

                h = h + h_cond
                h_cond = None
            hs.append(h)

            if self.use_scale_adjustment:
                h = h * self.input_scale[idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  + self.input_shift[idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        h = self.middle_block(h, emb, context, time_rel_pos_bias, focus_present_mask_only, selfattn_mode=selfattn_mode, attn_mode=attn_mode, slidewin=slidewin, reset=reset, second_weights=second_weights, drop_mask=drop_mask)
        if self.use_scale_adjustment:
            h = h * self.mid_scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + \
                self.mid_shift.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        for idx, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            if idx == len(self.output_blocks) -1:
                # import pdb; pdb.set_trace()
                h = module(h, emb, context, time_rel_pos_bias, focus_present_mask_only, return_single_branch='full', selfattn_mode=selfattn_mode, attn_mode=attn_mode, slidewin=slidewin, reset=reset, second_weights=second_weights, drop_mask=drop_mask)
            else:
                h = module(h, emb, context, time_rel_pos_bias, focus_present_mask_only, return_single_branch='full',  selfattn_mode=selfattn_mode, attn_mode=attn_mode, slidewin=slidewin, reset=reset, second_weights=second_weights, drop_mask=drop_mask)

            if self.use_scale_adjustment:
                h = h * self.output_scale[idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) +  self.output_shift[idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        h = h.type(x.dtype)
        h = self.out(h, emb, context, time_rel_pos_bias, focus_present_mask_only, return_single_branch='full',  selfattn_mode=selfattn_mode, attn_mode=attn_mode, slidewin=slidewin, reset=reset, second_weights=second_weights, drop_mask=drop_mask)

        if self.anchor_infer_once and self.counter > 0:
            h = torch.cat([h, torch.zeros((B, self.out_channels, self.frame_num-1, H, W)).to(h)], dim=2)

        h = h.permute(0,2,1,3,4).reshape(N, self.out_channels, H, W)

        if self.anchor_infer_once:
            self.counter = self.counter + 1

        return h


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from .util import (instantiate_from_config)

    print("load t2i model ... ")
    config_path = "./configs/sd_v2_base_ip_extra.yaml"
    config = OmegaConf.load(config_path)
    unet = instantiate_from_config(config.unet_config)
    x = None
    t = None

    output = unet(x, t)
    print(output.shape)
