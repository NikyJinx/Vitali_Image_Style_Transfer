import torch
import torch.nn as nn
import torch.nn.functional as F

# Definizione del modulo di attenzione alleggerito con il nome originale
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, reduced_channels=8, dropout_prob=0.5):
        super(AttentionLayer, self).__init__()
        self.query_key_conv = nn.Conv2d(in_channels, in_channels // reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // reduced_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query_key = self.query_key_conv(x)
        query = query_key.view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x (C/reduced_channels)
        key = query_key.view(batch_size, -1, width * height)  # B x (C/reduced_channels) x N
        attention = torch.bmm(query, key)  # B x N x N
        attention = F.softmax(attention, dim=-1)  # B x N x N
        value = self.value_conv(x).view(batch_size, -1, width * height)  # B x (C/reduced_channels) x N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x (C/reduced_channels) x N
        out = out.view(batch_size, C // reduced_channels, width, height)  # B x (C/reduced_channels) x W x H
        out = F.interpolate(out, size=(width, height), mode='bilinear', align_corners=True)  # Upsample to original size
        out = self.dropout(self.gamma * out) + x
        return out

# Definizione dei layer di convoluzione
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance", dropout_prob=0.5):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm_type = norm
        self.dropout = nn.Dropout(dropout_prob)
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        out = self.dropout(out)
        return out

# Definizione dei layer residui
class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, dropout_prob=0.5):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, dropout_prob=dropout_prob)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, dropout_prob=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return out

# Definizione dei layer di deconvoluzione
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance", dropout_prob=0.5):
        super(DeconvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)
        self.norm_type = norm
        self.dropout = nn.Dropout(dropout_prob)
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        out = self.dropout(out)
        return out

# Definizione della rete di trasformazione con il meccanismo di attenzione alleggerito
class TransformerNetworkWithAttention(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(TransformerNetworkWithAttention, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1, dropout_prob=dropout_prob),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2, dropout_prob=dropout_prob),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2, dropout_prob=dropout_prob),
            nn.ReLU()
        )
        self.AttentionBlock = AttentionLayer(128, dropout_prob=dropout_prob)
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3, dropout_prob=dropout_prob), 
            ResidualLayer(128, 3, dropout_prob=dropout_prob), 
            ResidualLayer(128, 3, dropout_prob=dropout_prob), 
            ResidualLayer(128, 3, dropout_prob=dropout_prob), 
            ResidualLayer(128, 3, dropout_prob=dropout_prob)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1, dropout_prob=dropout_prob),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1, dropout_prob=dropout_prob),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None", dropout_prob=dropout_prob)
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.AttentionBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

# Definizione della rete di trasformazione con Tanh e meccanismo di attenzione alleggerito
class TransformerNetworkTanhWithAttention(TransformerNetworkWithAttention):
    def __init__(self, tanh_multiplier=150, dropout_prob=0.5):
        super(TransformerNetworkTanhWithAttention, self).__init__(dropout_prob=dropout_prob)
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1, dropout_prob=dropout_prob),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1, dropout_prob=dropout_prob),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None", dropout_prob=dropout_prob),
            nn.Tanh()
        )
        self.tanh_multiplier = tanh_multiplier

    def forward(self, x):
        return super(TransformerNetworkTanhWithAttention, self).forward(x) * self.tanh_multiplier
