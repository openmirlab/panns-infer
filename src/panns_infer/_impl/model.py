"""Faithful evaluation-only PANNs CNN14, with checkpoint-owned frontend filters.

All parameter names retain the upstream checkpoint layout. STFT convolution
and transposed mel matrix are restored rather than regenerated with librosa.
Reads: normalized 32 kHz mono batches and the verified original state dictionary.
"""
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 32000
MIN_SAMPLES = 31 * 320
CLASSES = 527
EMBEDDING_DIM = 2048


class STFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.conv_imag = nn.Conv1d(1, 513, 1024, stride=320, bias=False)

    def forward(self, x):
        x = F.pad(x[:, None], (512, 512), mode='reflect')
        real = self.conv_real(x)[:, None].transpose(2, 3)
        imag = self.conv_imag(x)[:, None].transpose(2, 3)
        return real, imag


class Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFT()

    def forward(self, x):
        real, imag = self.stft(x)
        return real ** 2 + imag ** 2


class Logmel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.empty(64, 513).T, requires_grad=False)

    def forward(self, x):
        return 10.0 * torch.log10(torch.clamp(torch.matmul(x, self.melW), min=1e-10))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool_size)


class Cnn14(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram()
        self.logmel_extractor = Logmel()
        self.bn0 = nn.BatchNorm2d(64)
        channels = (1, 64, 128, 256, 512, 1024, 2048)
        for i in range(6):
            setattr(self, f'conv_block{i + 1}', ConvBlock(channels[i], channels[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, CLASSES)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['model'], strict=True)

    def forward(self, waveform):
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(6):
            pool = (2, 2) if i < 5 else (1, 1)
            x = getattr(self, f'conv_block{i + 1}')(x, pool)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        embedding = F.relu_(self.fc1(x))
        scores = torch.sigmoid(self.fc_audioset(embedding))
        return {'clipwise_output': scores, 'embedding': embedding}
