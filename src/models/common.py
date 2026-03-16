import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Convert 2D image to Patch Embeddings."""

    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernel_fn = nn.ELU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.kernel_fn(q) + 1
        k = self.kernel_fn(k) + 1

        q = q * self.scale
        kv = k.transpose(-2, -1) @ v
        x = q @ kv

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocusedLinearAttention(nn.Module):
    """
    Focused Linear Attention (ICCV 2023)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        focusing_factor=3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.focusing_factor = focusing_factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        q = q / q.sum(dim=-1, keepdim=True)
        k = k / k.sum(dim=-1, keepdim=True)
        q = q**self.focusing_factor
        k = k**self.focusing_factor
        q = q / q.sum(dim=-1, keepdim=True)
        k = k / k.sum(dim=-1, keepdim=True)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = out + self.dwc(out.transpose(1, 2)).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Standard Conv Block (Conv + BatchNorm + Activation)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv5 Bottleneck block with optional shortcut."""

    def __init__(
        self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(hidden_channels, out_channels, 3, 1, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Module(nn.Module):
    """YOLOv5 C3 Module (Cross Stage Partial Network)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottlenecks=3,
        shortcut=True,
        groups=1,
        expansion=0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels, hidden_channels, shortcut, groups, expansion
                )
                for _ in range(num_bottlenecks)
            ]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class InvertedResidual(nn.Module):
    """MobileNetV3-style block，比 C3 更适合文字特征提取"""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2.0, use_se=True):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch

        layers = []
        if hidden != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ])
        if use_se:
            layers.append(SEBlock(hidden, reduction=4))
        layers.extend([
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


def drop_path(
    x: torch.Tensor, drop_prob: float = 1.0, inplace: bool = False
) -> torch.Tensor:
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask: torch.Tensor = x.new_empty(mask_shape).bernoulli_(1-drop_prob)
    mask.div_(1-drop_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) regularization layer.
    During training, randomly drops entire residual paths with probability `p`.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
