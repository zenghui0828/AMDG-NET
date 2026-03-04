import torch
import torch.nn as nn
import math
from typing import Optional, List, Callable
from functools import partial


# ==============================
# 1. ResNet 系列
# ==============================

class BasicBlock(nn.Module):
    """ResNet 基本块（适用于 ResNet18/34）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈块（适用于 ResNet50及以上）"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet 主网络"""

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# ==============================
# 2. VGG16
# ==============================

class VGG(nn.Module):
    """VGG 网络"""

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG16(num_classes=1000, batch_norm=True):
    """VGG16 模型"""
    cfg = cfgs['D']
    model = VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes)
    return model


# ==============================
# 3. MobileNetV3
# ==============================

def _make_divisible(v, divisor=8, min_value=None):
    """
    确保所有层的通道数都能被 divisor 整除
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保不超过原值的10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=None,
                 activation_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class InvertedResidualConfig:
    def __init__(self,
                 input_channels,
                 kernel,
                 expanded_channels,
                 out_channels,
                 use_se,
                 activation,
                 stride):
        self.input_channels = input_channels
        self.kernel = kernel
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf,
                 norm_layer):
        super(InvertedResidual, self).__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = h_swish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels,
                                           cnf.expanded_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_channels,
                                       cnf.expanded_channels,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_channels,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SELayer(cnf.expanded_channels))

        # project
        layers.append(nn.Conv2d(cnf.expanded_channels,
                                cnf.out_channels,
                                kernel_size=1,
                                bias=False))
        layers.append(norm_layer(cnf.out_channels))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting,
                 last_channel,
                 num_classes=1000,
                 block=None,
                 norm_layer=None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, List) and
                  all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3,
                                       firstconv_output_channels,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=h_swish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels,
                                       lastconv_output_channels,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=h_swish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            h_swish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def mobilenet_v3_large(num_classes=1000, width_mult=1.0):
    inverted_residual_setting = [
        # input, kernel, expanded, out, use_se, activation, stride
        InvertedResidualConfig(16, 3, 16, 16, False, "RE", 1),
        InvertedResidualConfig(16, 3, 64, 24, False, "RE", 2),  # C1
        InvertedResidualConfig(24, 3, 72, 24, False, "RE", 1),
        InvertedResidualConfig(24, 5, 72, 40, True, "RE", 2),  # C2
        InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1),
        InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1),
        InvertedResidualConfig(40, 3, 240, 80, False, "HS", 2),  # C3
        InvertedResidualConfig(80, 3, 200, 80, False, "HS", 1),
        InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1),
        InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1),
        InvertedResidualConfig(80, 3, 480, 112, True, "HS", 1),
        InvertedResidualConfig(112, 3, 672, 112, True, "HS", 1),
        InvertedResidualConfig(112, 5, 672, 160, True, "HS", 2),  # C4
        InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1),
        InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1),
    ]
    last_channel = _make_divisible(1280 * max(1.0, width_mult))

    model = MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                        last_channel=last_channel,
                        num_classes=num_classes)
    return model


def mobilenet_v3_small(num_classes=1000, width_mult=1.0):
    inverted_residual_setting = [
        # input, kernel, expanded, out, use_se, activation, stride
        InvertedResidualConfig(16, 3, 16, 16, True, "RE", 2),  # C1
        InvertedResidualConfig(16, 3, 72, 24, False, "RE", 2),  # C2
        InvertedResidualConfig(24, 3, 88, 24, False, "RE", 1),
        InvertedResidualConfig(24, 5, 96, 40, True, "HS", 2),  # C3
        InvertedResidualConfig(40, 5, 240, 40, True, "HS", 1),
        InvertedResidualConfig(40, 5, 240, 40, True, "HS", 1),
        InvertedResidualConfig(40, 5, 120, 48, True, "HS", 1),
        InvertedResidualConfig(48, 5, 144, 48, True, "HS", 1),
        InvertedResidualConfig(48, 5, 288, 96, True, "HS", 2),  # C4
        InvertedResidualConfig(96, 5, 576, 96, True, "HS", 1),
        InvertedResidualConfig(96, 5, 576, 96, True, "HS", 1),
    ]
    last_channel = _make_divisible(1024 * max(1.0, width_mult))

    model = MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                        last_channel=last_channel,
                        num_classes=num_classes)
    return model


def MobileNetV3_Large(num_classes=1000):
    """MobileNetV3 Large 版本"""
    return mobilenet_v3_large(num_classes=num_classes)


def MobileNetV3_Small(num_classes=1000):
    """MobileNetV3 Small 版本"""
    return mobilenet_v3_small(num_classes=num_classes)


# ==============================
# 4. Vision Transformer (ViT)
# ==============================

class PatchEmbedding(nn.Module):
    """将图像分割为 patch 并嵌入"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """多层感知机（前馈网络）"""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 编码块"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer 主网络"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.0, embed_dropout=0.0):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(embed_dropout)

        # Transformer 编码器
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]  # 只返回 cls_token 的输出

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ViT 变体
def ViT_Tiny(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3,
        num_classes=num_classes
    )


def ViT_Small(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        num_classes=num_classes
    )


def ViT_Base(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes
    )


# ==============================
# 5. 模型工厂（方便选择）
# ==============================

def create_model(model_name, num_classes):
    """创建模型工厂"""
    model_dict = {
        # ResNet 系列
        'resnet18': ResNet18(num_classes),
        'resnet34': ResNet34(num_classes),
        'resnet50': ResNet50(num_classes),
        'resnet101': ResNet101(num_classes),
        'resnet152': ResNet152(num_classes),

        # VGG 系列
        'vgg16': VGG16(num_classes),

        # MobileNetV3 系列
        'mobilenetv3_large': MobileNetV3_Large(num_classes),
        'mobilenetv3_small': MobileNetV3_Small(num_classes),

        # Vision Transformer
        'vit_tiny': ViT_Tiny(num_classes),
        'vit_small': ViT_Small(num_classes),
        'vit_base': ViT_Base(num_classes),

        # 你的模型（需要从原文件导入）
        'efficientnetv2': None,  # 将在训练脚本中单独处理
    }

    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise ValueError(f"未知模型: {model_name}。可用模型: {list(model_dict.keys())}")


# ==============================
# 6. 计算复杂度分析工具
# ==============================

def analyze_model_complexity(model, input_size=(1, 3, 224, 224), device='cpu'):
    """分析模型的计算复杂度"""
    model = model.to(device)
    model.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算 FLOPs（使用 torchprofile 或手动计算）
    def count_flops():
        # 简化的 FLOPs 计算
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # 卷积层 FLOPs = 输出尺寸 * (核宽*核高*输入通道) * 输出通道 * 2
                h_out = (input_size[2] + 2 * module.padding[0] - module.dilation[0] * (
                        module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
                w_out = (input_size[3] + 2 * module.padding[1] - module.dilation[1] * (
                        module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
                flops += h_out * w_out * module.in_channels * module.out_channels * module.kernel_size[0] * \
                         module.kernel_size[1] * 2
            elif isinstance(module, nn.Linear):
                # 全连接层 FLOPs = 输入维度 * 输出维度 * 2
                flops += module.in_features * module.out_features * 2
        return flops

    try:
        flops = count_flops()
    except:
        flops = "无法计算"

    # 计算推理速度
    import time
    test_input = torch.randn(input_size).to(device)

    # 预热
    for _ in range(10):
        _ = model(test_input)

    # 正式测试
    times = []
    for _ in range(100):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        _ = model(test_input)
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.time() - start)

    avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
    fps = 1.0 / avg_time * 1000 if avg_time > 0 else 0

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'inference_time_ms': avg_time,
        'fps': fps
    }


if __name__ == '__main__':
    # 测试代码
    print("测试各模型...")

    # 测试 VGG16
    model = VGG16(num_classes=38)
    complexity = analyze_model_complexity(model)
    print("\nVGG16 复杂度分析:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")

    # 测试 MobileNetV3 Large
    model = MobileNetV3_Large(num_classes=38)
    complexity = analyze_model_complexity(model)
    print("\nMobileNetV3 Large 复杂度分析:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")

    # 测试 MobileNetV3 Small
    model = MobileNetV3_Small(num_classes=38)
    complexity = analyze_model_complexity(model)
    print("\nMobileNetV3 Small 复杂度分析:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")