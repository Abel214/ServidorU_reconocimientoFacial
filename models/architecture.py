import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_HEIGHT  = 112
IMG_WIDTH   = 92
NUM_CLASSES = 40

IMG_SIZE_SHUFFLE = 112
IMG_SIZE_GHOST = 112
class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=1, activation=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True) if activation else nn.Identity()
        )
    def forward(self, x):
        return self.block(x)


class DepthwiseBN(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=stride,
                      padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion):
        super().__init__()
        expanded = in_c * expansion
        self.use_residual = (stride == 1 and in_c == out_c)

        layers_list = []
        if expansion != 1:
            layers_list.append(ConvBN(in_c, expanded, 1, padding=0))
        layers_list += [
            DepthwiseBN(expanded, stride=stride),
            nn.Conv2d(expanded, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        self.block = nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

# ======================================================================
# BLOQUES BASE — versión ArcFace (PReLU + GroupNorm en stem)
# ======================================================================

class ConvBN_x2(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c)   # PReLU por canal en lugar de ReLU
        )
    def forward(self, x):
        return self.conv(x)


class DepthwiseBN_x2(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=stride,
                      padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels)
        )
    def forward(self, x):
        return self.dw(x)


class Bottleneck_x2(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion):
        super().__init__()
        mid_c = in_c * expansion
        self.block = nn.Sequential(
            # Expand
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            # Depthwise
            nn.Conv2d(mid_c, mid_c, 3, stride=stride,
                      padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            # Project (sin activación al final)
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.use_residual = (stride == 1 and in_c == out_c)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# ======================================================================
# ARQUITECTURA 1 — MobileFaceNet (CE, Experimentos 1 y 2)
# Input: (B, 1, 112, 92) → Output: (B, 40)
# ======================================================================

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet adaptado para ORL Dataset
    Input : (B, 1, 112, 92) — escala de grises
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, embedding_dim=128):
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            ConvBN(1, 64, 3, stride=2, padding=1),    # → (56, 46)
            DepthwiseBN(64, stride=1)
        )

        # ---- Bottleneck blocks ----
        # (expansion, out_channels, repeticiones, stride_primera)
        cfg = [
            (2,  64,  5, 2),   # → (28, 23)
            (4, 128,  1, 2),   # → (14, 12)
            (2, 128,  6, 1),   # → (14, 12)
            (4, 128,  1, 2),   # → ( 7,  6)
            (2, 128,  2, 1),   # → ( 7,  6)
        ]

        blocks = []
        in_c = 64
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(Bottleneck(in_c, c, stride, t))
                in_c = c
        self.bottlenecks = nn.Sequential(*blocks)

        # ---- Conv 1×1 ----
        self.conv_512 = ConvBN(128, 512, 1, padding=0)   # → (7, 6, 512)
        self.gdconv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(7, 6),
            groups=512, bias=False),       
            nn.GroupNorm(num_groups=32, num_channels=512) 
        )

        # ---- Embedding 1×1 ----
        self.embedding = nn.Sequential(
            nn.Conv2d(512, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()                                  # → 128-D
        )

        # ---- Clasificador ----
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.bottlenecks(x)
        x = self.conv_512(x)
        x = self.gdconv(x)
        x = self.embedding(x)
        return self.classifier(x)


# ======================================================================
# ARQUITECTURA 2 — MobileFaceNet2 (PReLU + ArcFace, Experimento 3)
# Input: (B, 1, 112, 92) → Output: embedding (B, 128)
# Sin classifier — ArcFaceHead lo reemplaza en inferencia
# ======================================================================

class MobileFaceNet2(nn.Module):
    """
    MobileFaceNet con PReLU + ArcFace
    El classifier se elimina — ArcFaceLoss lo reemplaza
    """
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Stem con stride=1 en primera conv (fiel al paper)
        self.stem = nn.Sequential(
            ConvBN_x2(1, 64, 3, stride=1, padding=1),   # stride=1, preserva resolución
            DepthwiseBN_x2(64, stride=1)
        )

        cfg = [
            (2,  64,  5, 2),
            (4, 128,  1, 2),
            (2, 128,  6, 1),
            (4, 128,  1, 2),
            (2, 128,  2, 1),
        ]

        blocks = []
        in_c = 64
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(Bottleneck_x2(in_c, c, stride, t))
                in_c = c
        self.bottlenecks = nn.Sequential(*blocks)

        self.conv_512 = ConvBN_x2(128, 512, 1, padding=0)

        self.gdconv = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(14, 12), groups=512, bias=False),
        nn.GroupNorm(num_groups=32, num_channels=512)
        )

        self.embedding = nn.Sequential(
            nn.Conv2d(512, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()
        )

        self.dropout = nn.Dropout(p=0.4)
        # Sin self.classifier — ArcFace lo maneja

    def forward(self, x):
        x = self.stem(x)
        x = self.bottlenecks(x)
        x = self.conv_512(x)
        x = self.gdconv(x)
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x   # retorna embedding 128-D normalizado


# ======================================================================
# CABEZA ARCFACE — para inferencia con similitud coseno
# ======================================================================

class ArcFaceHead(nn.Module):
    """
    Guarda los pesos del classifier ArcFace (shape: num_classes × embedding_dim).
    En inferencia calcula similitud coseno embedding vs centros de clase.
    """
    def __init__(self, embedding_dim=128, num_classes=NUM_CLASSES, s=30.0):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        emb_norm    = F.normalize(embedding, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cosine      = emb_norm @ weight_norm.T  # (B, num_classes)
        return cosine * self.s
    
class ArcFaceHead_Mobile(nn.Module):
    """
    Guarda los pesos del classifier ArcFace (shape: num_classes × embedding_dim).
    En inferencia calcula similitud coseno embedding vs centros de clase.
    """
    def __init__(self, embedding_dim=128, num_classes=NUM_CLASSES, s=30.0):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        emb_norm    = F.normalize(embedding, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cosine      = emb_norm @ weight_norm.T  # (B, num_classes)
        return cosine * self.s
    

# ======================================================================
# BLOQUES BASE — ShuffleFaceNet (ShuffleNetV2 + Dense + PReLU)
# ======================================================================
 
class ConvBNPReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleDenseBlock(nn.Module):
    """Bloque tipo DenseNet + ShuffleNetV2 para etapas 2-4."""
    def __init__(self, in_c, stride=1):
        super().__init__()
        self.stride = stride
        branch_c = in_c // 2

        if stride == 1:
            self.branch1 = nn.Identity()
            branch1_out = branch_c
            branch2_in = branch_c
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, in_c, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.PReLU(in_c)
            )
            branch1_out = in_c
            branch2_in = in_c

        mid_c = branch2_in * 2
        self.branch2 = nn.Sequential(
            ConvBNPReLU(branch2_in, mid_c, 1),
            nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            ConvBNPReLU(mid_c, branch2_in, 1)
        )

        self.out_channels = branch1_out + branch2_in

    def forward(self, x):
        if self.stride == 1:
            c = x.size(1) // 2
            x1, x2 = x[:, :c], x[:, c:]
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)

        out = channel_shuffle(out, groups=2)
        return out

 
 
# ======================================================================
# ARQUITECTURA 3 — ShuffleFaceNet (CE, Experimentos 1 y 2)
# Input: (B, 1, 112, 112) → Output: (B, 40)
# ======================================================================
class ShuffleFaceNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embedding_dim=128, dropout_p=0.4):
        super().__init__()

        # Stem: conv 3x3 stride=2 -> 112x112 -> 56x56
        # Reducido de 24 a 16 canales iniciales (modelo más pequeño para dataset chico)
        self.stem = ConvBNPReLU(1, 16, 3, stride=2, padding=1)

        # Etapa 1: 56x56 -> 28x28
        self.stage1 = nn.Sequential(ShuffleDenseBlock(16, stride=2))
        c1 = self.stage1[-1].out_channels  # 32

        # Etapa 2: 28x28 -> 14x14 (4 -> 3 bloques)
        stage2_blocks = [ShuffleDenseBlock(c1, stride=2)]
        c2 = stage2_blocks[-1].out_channels
        for _ in range(2):
            stage2_blocks.append(ShuffleDenseBlock(c2, stride=1))
            c2 = stage2_blocks[-1].out_channels
        self.stage2 = nn.Sequential(*stage2_blocks)

        # Etapa 3: 14x14 -> 7x7 (8 -> 4 bloques)
        stage3_blocks = [ShuffleDenseBlock(c2, stride=2)]
        c3 = stage3_blocks[-1].out_channels
        for _ in range(3):
            stage3_blocks.append(ShuffleDenseBlock(c3, stride=1))
            c3 = stage3_blocks[-1].out_channels
        self.stage3 = nn.Sequential(*stage3_blocks)

        # Etapa 4: 7x7 -> 7x7 (3 -> 2 bloques)
        stage4_blocks = []
        c4 = c3
        for _ in range(2):
            stage4_blocks.append(ShuffleDenseBlock(c4, stride=1))
            c4 = stage4_blocks[-1].out_channels
        self.stage4 = nn.Sequential(*stage4_blocks)

        # Conv 1x1 expansión - reducido de 512 a 256
        self.conv_expand = ConvBNPReLU(c4, 256, 1)

        # GDC: Global Depthwise Conv 7x7 -> reduce a (1,1)
        self.gdc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, groups=256, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256)
        )

        # Embedding 128-D
        self.embedding = nn.Sequential(
            nn.Conv2d(256, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()
        )

        # Dropout antes del classifier (regularización contra overfitting)
        self.dropout = nn.Dropout(p=dropout_p)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_expand(x)
        x = self.gdc(x)
        x = self.embedding(x)
        x = self.dropout(x)
        return self.classifier(x)
 
 
# ======================================================================
# ARQUITECTURA 4 — ShuffleFaceNet2_Arface (PReLU + ArcFace, Experimento 3)
# Input: (B, 1, 112, 112) → Output: embedding L2-normalizado (B, 128)
# ======================================================================
 
class ConvBNPReLU_x2(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def channel_shuffle_x2(x, groups=2):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleDenseBlock_x2(nn.Module):
    def __init__(self, in_c, stride=1):
        super().__init__()
        self.stride = stride
        branch_c = in_c // 2

        if stride == 1:
            self.branch1 = nn.Identity()
            branch1_out  = branch_c
            branch2_in   = branch_c
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, in_c, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.PReLU(in_c)
            )
            branch1_out = in_c
            branch2_in  = in_c

        mid_c = branch2_in * 2
        self.branch2 = nn.Sequential(
            ConvBNPReLU(branch2_in, mid_c, 1),
            nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            ConvBNPReLU(mid_c, branch2_in, 1)
        )
        self.out_channels = branch1_out + branch2_in

    def forward(self, x):
        if self.stride == 1:
            c = x.size(1) // 2
            x1, x2 = x[:, :c], x[:, c:]
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle_x2(out, groups=2)





class ShuffleFaceNet2_Arface(nn.Module):
    """
    Idéntica a ShuffleFaceNet pero:
      - Sin classifier lineal final
      - Embedding L2-normalizado (para ArcFace)
      - Sin Dropout (el margen angular ya regulariza)
      - Inicialización Xavier Normal en Conv y Linear
    """
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.stem = ConvBNPReLU_x2(1, 16, 3, stride=2, padding=1)

        self.stage1 = nn.Sequential(ShuffleDenseBlock_x2(16, stride=2))
        c1 = self.stage1[-1].out_channels

        stage2 = [ShuffleDenseBlock_x2(c1, stride=2)]
        c2 = stage2[-1].out_channels
        for _ in range(2):
            stage2.append(ShuffleDenseBlock_x2(c2, stride=1))
            c2 = stage2[-1].out_channels
        self.stage2 = nn.Sequential(*stage2)

        stage3 = [ShuffleDenseBlock_x2(c2, stride=2)]
        c3 = stage3[-1].out_channels
        for _ in range(3):
            stage3.append(ShuffleDenseBlock_x2(c3, stride=1))
            c3 = stage3[-1].out_channels
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        c4 = c3
        for _ in range(2):
            stage4.append(ShuffleDenseBlock_x2(c4, stride=1))
            c4 = stage4[-1].out_channels
        self.stage4 = nn.Sequential(*stage4)

        self.conv_expand = ConvBNPReLU_x2(c4, 256, 1)

        self.gdc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, groups=256, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256)
        )

        # Proyección a embedding — SIN activación (lineal puro)
        self.embedding_proj = nn.Sequential(
            nn.Conv2d(256, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()
        )

        # Inicialización Xavier Normal
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_expand(x)
        x = self.gdc(x)
        x = self.embedding_proj(x)
        # L2-normalización: necesaria para ArcFace
        x = F.normalize(x, p=2, dim=1)
        return x   # shape: (B, 128) — embeddings normalizados







# ---------------------------------------------------------------------------
# Bloques del modelo — GhostFaceNetV2
# ---------------------------------------------------------------------------
class ConvBNPReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.PReLU(out_c)
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
 
def _make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
 
 
class SqueezeExcite(nn.Module):
    """SE con PReLU en la reduccion."""
    def __init__(self, in_chs, se_ratio=0.25):
        super().__init__()
        reduced_chs = _make_divisible(int(in_chs * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act = nn.PReLU(reduced_chs)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
 
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.act(self.conv_reduce(x_se))
        x_se = self.conv_expand(x_se)
        return x * torch.sigmoid(x_se)
 
 
class DFCAttention(nn.Module):
    def __init__(self, chs, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_h = nn.Conv2d(chs, chs, kernel_size=(1, kernel_size),
                                 padding=(0, pad), groups=chs, bias=False)
        self.conv_v = nn.Conv2d(chs, chs, kernel_size=(kernel_size, 1),
                                 padding=(pad, 0), groups=chs, bias=False)
        self.bn = nn.BatchNorm2d(chs)
 
    def forward(self, x):
        h, w = x.shape[-2:]
        att = self.pool(x)
        att = self.conv_h(att)
        att = self.conv_v(att)
        att = self.bn(att)
        att = torch.sigmoid(att)
        att = F.interpolate(att, size=(h, w), mode="nearest")
        return att



class GhostModule(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size=1, ratio=2, dw_size=3,
                 stride=1, use_act=True):
        super().__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)
 
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            nn.PReLU(init_chs) if use_act else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size // 2,
                      groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            nn.PReLU(new_chs) if use_act else nn.Identity(),
        )
 
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :, :]
 
 
class GhostBottleneckV2(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, se_ratio=0.0, use_dfc=True):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.use_dfc = use_dfc

        if use_dfc:
            self.dfc = DFCAttention(mid_chs)          # <-- antes: DFCAttention(in_chs)

        self.ghost1 = GhostModule(in_chs, mid_chs, use_act=True)

        if stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size,
                                      stride=stride,
                                      padding=(dw_kernel_size - 1) // 2,
                                      groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None

        self.se = SqueezeExcite(mid_chs, se_ratio) if has_se else None

        self.ghost2 = GhostModule(mid_chs, out_chs, use_act=False)

        if in_chs == out_chs and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2,
                          groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
        self.out_channels = out_chs

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)

        if self.conv_dw is not None:
            x1 = self.bn_dw(self.conv_dw(x1))

        if self.use_dfc:
            att = self.dfc(x1)                      
            if att.shape[-2:] != x1.shape[-2:]:
                att = F.interpolate(att, size=x1.shape[-2:], mode="nearest")
            x1 = x1 * att

        if self.se is not None:
            x1 = self.se(x1)

        x1 = self.ghost2(x1)
        return x1 + self.shortcut(residual)




# ---------------------------------------------------------------------------
# Modelo — GhostFaceNetV2
# ---------------------------------------------------------------------------
class GhostFaceNetV2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embedding_dim=128,
                 dropout_p=0.4, use_dfc=True):
        super().__init__()
 
        # Stem: conv 3x3 stride=2 -> 112x112 -> 56x56
        self.stem = ConvBNPReLU(1, 16, 3, stride=2, padding=1)
 
        # Etapa 1: 56x56 -> 28x28
        self.stage1 = nn.Sequential(
            GhostBottleneckV2(16, mid_chs=48, out_chs=24, dw_kernel_size=3,
                               stride=2, se_ratio=0.0, use_dfc=use_dfc)
        )
        c1 = self.stage1[-1].out_channels  # 24
 
        # Etapa 2: 28x28 -> 14x14 (1 stride=2 + 2 stride=1)
        stage2_blocks = [
            GhostBottleneckV2(c1, mid_chs=72, out_chs=40, dw_kernel_size=5,
                               stride=2, se_ratio=0.25, use_dfc=use_dfc)
        ]
        c2 = stage2_blocks[-1].out_channels
        for _ in range(2):
            stage2_blocks.append(
                GhostBottleneckV2(c2, mid_chs=120, out_chs=40, dw_kernel_size=5,
                                   stride=1, se_ratio=0.25, use_dfc=use_dfc)
            )
            c2 = stage2_blocks[-1].out_channels
        self.stage2 = nn.Sequential(*stage2_blocks)
 
        # Etapa 3: 14x14 -> 7x7 (1 stride=2 + 3 stride=1)
        stage3_blocks = [
            GhostBottleneckV2(c2, mid_chs=240, out_chs=80, dw_kernel_size=3,
                               stride=2, se_ratio=0.0, use_dfc=use_dfc)
        ]
        c3 = stage3_blocks[-1].out_channels
        mids3 = [200, 184, 184]
        se3 = [0.0, 0.0, 0.25]
        for mid, se in zip(mids3, se3):
            stage3_blocks.append(
                GhostBottleneckV2(c3, mid_chs=mid, out_chs=80, dw_kernel_size=3,
                                   stride=1, se_ratio=se, use_dfc=use_dfc)
            )
            c3 = stage3_blocks[-1].out_channels
        self.stage3 = nn.Sequential(*stage3_blocks)
 
        # Etapa 4: 7x7 -> 7x7 (2 bloques stride=1)
        stage4_blocks = []
        c4 = c3
        for mid in [480, 672]:
            stage4_blocks.append(
                GhostBottleneckV2(c4, mid_chs=mid, out_chs=112, dw_kernel_size=3,
                                   stride=1, se_ratio=0.25, use_dfc=use_dfc)
            )
            c4 = stage4_blocks[-1].out_channels
        self.stage4 = nn.Sequential(*stage4_blocks)
 
        # Conv 1x1 expansión
        self.conv_expand = ConvBNPReLU(c4, 256, 1)
 
        # GDC: Global Depthwise Conv 7x7 -> reduce a (1,1)
        self.gdc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, groups=256, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256)
        )
 
        # Embedding 128-D
        self.embedding = nn.Sequential(
            nn.Conv2d(256, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()
        )
 
        # Dropout antes del classifier (regularización contra overfitting)
        self.dropout = nn.Dropout(p=dropout_p)
 
        self.classifier = nn.Linear(embedding_dim, num_classes)
 
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_expand(x)
        x = self.gdc(x)
        x = self.embedding(x)
        x = self.dropout(x)
        return self.classifier(x)



class GhostFaceNetV2_ArcFace(nn.Module):
    """
    Idéntica a GhostFaceNetV2 pero:
      - Sin classifier lineal final
      - Embedding L2-normalizado (para ArcFace)
      - Sin Dropout
      - `use_dfc` controla si se aplican las atenciones DFC
    """
    def __init__(self, embedding_dim=128, use_dfc=True):
        super().__init__()

        self.stem = ConvBNPReLU(1, 16, 3, stride=2, padding=1)

        self.stage1 = nn.Sequential(
            GhostBottleneckV2(16, mid_chs=48, out_chs=24, dw_kernel_size=3,
                              stride=2, se_ratio=0.0, use_dfc=use_dfc)
        )
        c1 = self.stage1[-1].out_channels

        stage2 = [
            GhostBottleneckV2(c1, mid_chs=72, out_chs=40, dw_kernel_size=5,
                              stride=2, se_ratio=0.25, use_dfc=use_dfc)
        ]
        c2 = stage2[-1].out_channels
        for _ in range(2):
            stage2.append(
                GhostBottleneckV2(c2, mid_chs=120, out_chs=40, dw_kernel_size=5,
                                  stride=1, se_ratio=0.25, use_dfc=use_dfc)
            )
            c2 = stage2[-1].out_channels
        self.stage2 = nn.Sequential(*stage2)

        stage3 = [
            GhostBottleneckV2(c2, mid_chs=240, out_chs=80, dw_kernel_size=3,
                              stride=2, se_ratio=0.0, use_dfc=use_dfc)
        ]
        c3 = stage3[-1].out_channels
        mids3 = [200, 184, 184]
        se3   = [0.0, 0.0, 0.25]
        for mid, se in zip(mids3, se3):
            stage3.append(
                GhostBottleneckV2(c3, mid_chs=mid, out_chs=80, dw_kernel_size=3,
                                  stride=1, se_ratio=se, use_dfc=use_dfc)
            )
            c3 = stage3[-1].out_channels
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        c4 = c3
        for mid in [480, 672]:
            stage4.append(
                GhostBottleneckV2(c4, mid_chs=mid, out_chs=112, dw_kernel_size=3,
                                  stride=1, se_ratio=0.25, use_dfc=use_dfc)
            )
            c4 = stage4[-1].out_channels
        self.stage4 = nn.Sequential(*stage4)

        self.conv_expand = ConvBNPReLU(c4, 256, 1)

        self.gdc = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, groups=256, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256)
        )

        self.embedding_proj = nn.Sequential(
            nn.Conv2d(256, embedding_dim, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.Flatten()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_expand(x)
        x = self.gdc(x)
        x = self.embedding_proj(x)
        return F.normalize(x, p=2, dim=1)