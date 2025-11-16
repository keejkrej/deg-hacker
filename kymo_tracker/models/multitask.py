"""Model definitions for the kymo-tracker multi-task network."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional block with two conv layers, batch norm, ReLU, and optional dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoiseUNet(nn.Module):
    """UNet that only predicts denoising residuals."""

    def __init__(
        self,
        base_channels: int = 48,
        use_bn: bool = True,
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, use_bn=use_bn, dropout=encoder_dropout)
        self.enc2 = ConvBlock(
            base_channels,
            base_channels * 2,
            use_bn=use_bn,
            dropout=encoder_dropout,
        )
        self.enc3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            use_bn=use_bn,
            dropout=encoder_dropout,
        )
        self.down = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.bottleneck = ConvBlock(
            base_channels * 4,
            base_channels * 8,
            use_bn=use_bn,
            dropout=dropout,
        )
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        self.dec3 = ConvBlock(
            base_channels * 8,
            base_channels * 4,
            use_bn=use_bn,
            dropout=decoder_dropout,
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        self.dec2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
            use_bn=use_bn,
            dropout=decoder_dropout,
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        self.dec1 = ConvBlock(
            base_channels * 2,
            base_channels,
            use_bn=use_bn,
            dropout=decoder_dropout,
        )
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = torch.nn.functional.interpolate(
                d3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = torch.nn.functional.interpolate(
                d2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = torch.nn.functional.interpolate(
                d1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class TemporalLocator(nn.Module):
    """Lightweight temporal locator with 1D ViT-style attention."""

    def __init__(
        self,
        in_channels: int = 1,
        spatial_channels: int = 96,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_tracks: int = 3,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(spatial_channels, token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(token_dim, max_tracks * 2)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.spatial_encoder(x)
        pooled = features.mean(dim=-1)
        tokens = pooled.permute(0, 2, 1)
        tokens = self.proj(tokens)
        seq_len = tokens.size(1)
        if seq_len > self.pos_embed.size(1):
            raise ValueError(
                "TemporalLocator received sequence of length "
                f"{seq_len} but positional embedding supports up to "
                f"{self.pos_embed.size(1)}"
            )
        tokens = tokens + self.pos_embed[:, :seq_len]
        encoded = self.transformer(tokens)
        preds = self.head(encoded)
        preds = preds.view(encoded.size(0), seq_len, self.max_tracks, 2)
        preds = preds.permute(0, 2, 3, 1)
        center_logits = preds[:, :, 0]
        width_logits = preds[:, :, 1]
        centers = torch.sigmoid(center_logits)
        widths = torch.nn.functional.softplus(width_logits) + 1e-3
        return centers, widths


class MultiTaskUNet(nn.Module):
    """Wrapper that couples denoising U-Net with a temporal locator."""

    def __init__(
        self,
        base_channels: int = 48,
        use_bn: bool = True,
        max_tracks: int = 3,
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        locator_token_dim: int = 128,
        locator_layers: int = 2,
        locator_heads: int = 4,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.denoiser = DenoiseUNet(
            base_channels=base_channels,
            use_bn=use_bn,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )
        self.locator = TemporalLocator(
            in_channels=1,
            spatial_channels=base_channels * 2,
            token_dim=locator_token_dim,
            num_layers=locator_layers,
            num_heads=locator_heads,
            max_tracks=max_tracks,
            max_tokens=512,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = self.denoiser(x)
        denoised = torch.clamp(x - predicted_noise, 0.0, 1.0)
        centers, widths = self.locator(denoised.detach())
        return predicted_noise, centers, widths


__all__ = [
    "ConvBlock",
    "DenoiseUNet",
    "TemporalLocator",
    "MultiTaskUNet",
]
