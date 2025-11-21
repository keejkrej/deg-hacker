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


class Conv1DBlock(nn.Module):
    """1D Convolutional block with two conv layers, batch norm, ReLU, and optional dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    """1D U-Net for keypoint detection on 1D heatmaps."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Encoder
        self.enc1 = Conv1DBlock(in_channels, base_channels, use_bn=use_bn, dropout=dropout)
        self.enc2 = Conv1DBlock(base_channels, base_channels * 2, use_bn=use_bn, dropout=dropout)
        self.enc3 = Conv1DBlock(base_channels * 2, base_channels * 4, use_bn=use_bn, dropout=dropout)
        
        self.down = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = Conv1DBlock(base_channels * 4, base_channels * 8, use_bn=use_bn, dropout=dropout)
        
        # Decoder
        self.up3 = nn.ConvTranspose1d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = Conv1DBlock(base_channels * 8, base_channels * 4, use_bn=use_bn, dropout=dropout)
        
        self.up2 = nn.ConvTranspose1d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = Conv1DBlock(base_channels * 4, base_channels * 2, use_bn=use_bn, dropout=dropout)
        
        self.up1 = nn.ConvTranspose1d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = Conv1DBlock(base_channels * 2, base_channels, use_bn=use_bn, dropout=dropout)
        
        # Output head: single channel heatmap
        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))

        d3 = self.up3(b)
        if d3.shape[-1] != e3.shape[-1]:
            d3 = torch.nn.functional.interpolate(
                d3,
                size=e3.shape[-1],
                mode="linear",
                align_corners=False,
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = torch.nn.functional.interpolate(
                d2,
                size=e2.shape[-1],
                mode="linear",
                align_corners=False,
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = torch.nn.functional.interpolate(
                d1,
                size=e1.shape[-1],
                mode="linear",
                align_corners=False,
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class HeatmapLocator(nn.Module):
    """2D heatmap-based locator: treats keypoint detection as heatmap regression on full 16x512 slice."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        use_bn: bool = True,
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Use 2D U-Net to process full temporal-spatial slice at once
        # This allows the model to leverage temporal correlations
        self.unet2d = DenoiseUNet(
            base_channels=base_channels,
            use_bn=use_bn,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input and output heatmap.
        
        Args:
            x: (batch, 1, time=16, space=512) input kymograph slice
            
        Returns:
            heatmap: (batch, 1, time=16, space=512) predicted heatmap
        """
        # Process full 2D slice at once with 2D U-Net
        # The DenoiseUNet already handles (batch, channels, time, space) format
        # We'll use it to predict heatmap instead of noise
        heatmap = self.unet2d(x)
        
        # Ensure non-negative output (heatmaps should be >= 0)
        heatmap = torch.clamp(heatmap, min=0.0)
        
        return heatmap


class TemporalLocator(nn.Module):
    """Simple CNN locator: extract features, regress centers/widths directly using 1D conv."""

    def __init__(
        self,
        in_channels: int = 1,
        spatial_channels: int = 48,
        max_tracks: int = 3,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )
        
        # Process each time frame: use 1D conv along spatial dimension to regress directly
        self.regressor = nn.Sequential(
            nn.Conv1d(spatial_channels, spatial_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(spatial_channels, max_tracks * 2, kernel_size=1),
        )
        
        nn.init.xavier_uniform_(self.regressor[-1].weight, gain=0.1)
        nn.init.constant_(self.regressor[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 1, time=16, space=512)
        features = self.features(x)  # (batch, channels=48, time=16, space=512)
        
        # Process each time frame independently with 1D conv
        batch_size, channels, time_frames, spatial_pixels = features.shape
        # Reshape: (batch, channels, time, space) -> (batch*time, channels, space)
        features_reshaped = features.permute(0, 2, 1, 3).contiguous()  # (batch, time, channels, space)
        features_reshaped = features_reshaped.view(batch_size * time_frames, channels, spatial_pixels)
        
        # Regress: (batch*time, channels, space) -> (batch*time, max_tracks*2, space)
        preds = self.regressor(features_reshaped)  # (batch*time, max_tracks*2=6, space=512)
        
        # Reshape back: (batch*time, max_tracks*2, space) -> (batch, time, max_tracks*2, space)
        preds = preds.view(batch_size, time_frames, self.max_tracks * 2, spatial_pixels)
        preds = preds.permute(0, 2, 1, 3)  # (batch, max_tracks*2, time, space)
        
        # Split predictions
        center_logits = preds[:, :self.max_tracks, :, :]  # (batch, max_tracks=3, time=16, space=512)
        width_logits = preds[:, self.max_tracks:, :, :]   # (batch, max_tracks=3, time=16, space=512)
        
        # Centers: argmax over spatial dimension
        centers = torch.argmax(center_logits, dim=-1).float()  # (batch, max_tracks, time)
        centers = centers / (spatial_pixels - 1)  # Normalize to [0, 1]
        
        # Widths: max over spatial dimension
        widths = torch.nn.functional.softplus(width_logits).max(dim=-1)[0] + 1e-3  # (batch, max_tracks, time)
        widths = widths / spatial_pixels  # Normalize
        
        return centers, widths


class MultiTaskUNet(nn.Module):
    """Wrapper that couples denoising U-Net with a heatmap-based locator."""

    def __init__(
        self,
        base_channels: int = 48,
        use_bn: bool = True,
        max_tracks: int = 3,
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        locator_base_channels: int = 32,
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
        self.locator = HeatmapLocator(
            in_channels=1,
            base_channels=locator_base_channels,
            use_bn=use_bn,
            dropout=dropout,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, time=16, space=512) input kymograph slice
            
        Returns:
            predicted_noise: (batch, 1, time=16, space=512) predicted noise
            heatmap: (batch, 1, time=16, space=512) predicted heatmap
        """
        predicted_noise = self.denoiser(x)
        denoised = torch.clamp(x - predicted_noise, 0.0, 1.0)
        heatmap = self.locator(denoised.detach())
        return predicted_noise, heatmap


__all__ = [
    "ConvBlock",
    "Conv1DBlock",
    "DenoiseUNet",
    "UNet1D",
    "TemporalLocator",
    "HeatmapLocator",
    "MultiTaskUNet",
]
