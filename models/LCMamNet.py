import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SBA(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        mid_c = max(8, channel // reduction)
        self.mlp_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, mid_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, channel, 1, bias=False),
        )
        self.mlp_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, mid_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        att_sum = self.mlp_x(x) + self.mlp_g(g)
        return x * self.sigmoid(att_sum)


class SBAUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        skip_c = in_c // 2
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sba = SBA(channel=skip_c)
        self.conv = nn.Sequential(
            ConvBlock(in_c, out_c, k=1, p=0),
            ConvBlock(out_c, out_c, k=3, p=1),
        )

    def forward(self, x, skip):
        x_up = self.up(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=True)
        skip = self.sba(g=x_up, x=skip)
        return self.conv(torch.cat([x_up, skip], dim=1))


class CDConv(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        assert k >= 3 and k % 2 == 1
        branch_c = [out_c // 4] * 4
        for i in range(out_c - sum(branch_c)):
            branch_c[i] += 1
        pad = k - 1
        self.left = nn.Sequential(
            nn.ZeroPad2d((pad, 0, 0, 0)),
            nn.Conv2d(in_c, branch_c[0], kernel_size=(1, k), bias=False),
        )
        self.right = nn.Sequential(
            nn.ZeroPad2d((0, pad, 0, 0)),
            nn.Conv2d(in_c, branch_c[1], kernel_size=(1, k), bias=False),
        )
        self.top = nn.Sequential(
            nn.ZeroPad2d((0, 0, pad, 0)),
            nn.Conv2d(in_c, branch_c[2], kernel_size=(k, 1), bias=False),
        )
        self.bottom = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, pad)),
            nn.Conv2d(in_c, branch_c[3], kernel_size=(k, 1), bias=False),
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([self.left(x), self.right(x), self.top(x), self.bottom(x)], dim=1)
        x = self.act(self.bn(x))
        return self.fuse(x)


class CDConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = CDConv(in_c, out_c, k=3)

    def forward(self, x):
        return self.conv(x)


class TokenProjector(nn.Module):
    def __init__(self, in_c, embed_dim):
        super().__init__()
        self.pw = nn.Conv2d(in_c, embed_dim, 1, bias=False)
        self.dw = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pw(x)
        x = x + self.dw(x)
        return self.act(self.bn(x))


class BiMambaPreNorm(nn.Module):
    def __init__(self, dim, gamma_init=0.1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gamma = nn.Parameter(torch.full((dim,), gamma_init, dtype=torch.float32))

    def forward(self, x):
        y = self.norm(x)
        y_fwd = self.mamba_fwd(y)
        y_rev = torch.flip(y, dims=[1])
        y_bwd = self.mamba_bwd(y_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])
        y = 0.5 * (y_fwd + y_bwd)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return x + self.gamma * y


class LCR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 2
        self.proj_in = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.dwconv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dwconv5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.proj_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        x_in = self.proj_in(x)
        x3, x5 = x_in.chunk(2, dim=1)
        x3 = self.dwconv3x3(x3)
        x5 = self.dwconv5x5(x5)
        x_out = self.proj_out(self.act(torch.cat([x3, x5], dim=1)))
        return x + x_out


class BiMambaMultiScaleFusionStageA(nn.Module):
    def __init__(self, in_dims, embed_dim=72):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c)) for c in in_dims]
        )
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1)
        self.refiners = nn.ModuleList([LCR(embed_dim) for _ in in_dims])

    def forward(self, xs):
        batch = xs[0].shape[0]
        tokens = []
        shapes = []
        for i, x in enumerate(xs):
            feat = self.projs_in[i](x)
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            tokens.append(feat.flatten(2).transpose(1, 2))

        seq = self.mixer(torch.cat(tokens, dim=1))
        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            feat = seq[:, cursor : cursor + length, :]
            cursor += length
            feat = feat.transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            feat = self.refiners[i](feat)
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.projs_out[i](feat)
            outputs.append(torch.nan_to_num(out + xs[i], nan=0.0, posinf=1e4, neginf=-1e4))
        return outputs


class BiMambaMultiScaleFusionStageB(nn.Module):
    def __init__(self, in_dims, embed_dim=40):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c)) for c in in_dims]
        )
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1)
        self.refiners = nn.ModuleList([LCR(embed_dim) for _ in in_dims])

    def forward(self, xs):
        batch = xs[0].shape[0]
        tokens = []
        shapes = []
        for i, x in enumerate(xs):
            feat = self.projs_in[i](x)
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            tokens.append(feat.flatten(2).transpose(1, 2))

        seq = self.mixer(torch.cat(tokens, dim=1))
        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            feat = seq[:, cursor : cursor + length, :]
            cursor += length
            feat = feat.transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            feat = self.refiners[i](feat)
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.projs_out[i](feat)
            outputs.append(torch.nan_to_num(out + xs[i], nan=0.0, posinf=1e4, neginf=-1e4))
        return outputs


class CDBR_Block(nn.Module):
    def __init__(self, in_c, out_c, bottleneck):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
        )
        self.conv = CDConv(bottleneck, bottleneck, k=3)
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.reduce(x)
        x = self.conv(x)
        x = self.expand(x)
        return self.act(x + residual)


def _scaled_bottleneck(out_c):
    bottleneck = int(round(out_c * 0.625 / 8.0) * 8)
    return max(16, bottleneck)


class LCMamNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_c=32):
        super().__init__()
        self.inc = nn.Sequential(CDConvBlock(n_channels, base_c), CDConvBlock(base_c, base_c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = CDBR_Block(base_c, base_c * 2, bottleneck=_scaled_bottleneck(base_c * 2))
        self.stage2 = CDBR_Block(base_c * 2, base_c * 4, bottleneck=_scaled_bottleneck(base_c * 4))
        self.stage3 = CDBR_Block(base_c * 4, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))
        self.stage4 = CDBR_Block(base_c * 8, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))

        self.stageA = BiMambaMultiScaleFusionStageA(
            [base_c * 2, base_c * 4, base_c * 8, base_c * 8],
            embed_dim=72,
        )
        self.stageB = BiMambaMultiScaleFusionStageB(
            [base_c * 4, base_c * 8, base_c * 8],
            embed_dim=40,
        )

        self.up4 = SBAUpBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = SBAUpBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = SBAUpBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = SBAUpBlock(base_c + base_c, base_c)
        self.outc = nn.Conv2d(base_c, n_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        a1, a2, a3, a4 = self.stageA([x1, x2, x3, x4])
        b2, b3, b4 = self.stageB([a2, a3, a4])

        d4 = self.up4(b4, b3)
        d3 = self.up3(d4, b2)
        d2 = self.up2(d3, a1)
        out_feat = self.up1(d2, x0)
        return self.outc(out_feat)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LCMamNet().to(device).eval()
    x = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)
    print(tuple(y.shape))
