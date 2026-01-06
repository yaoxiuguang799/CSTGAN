import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from typing import Union

# ============================================================
# Loss Functions
# ============================================================
def cal_mae(pred, gt, mask=None):
    if mask is None:
        return torch.mean(torch.abs(pred - gt))
    return torch.sum(torch.abs(pred - gt) * mask) / (mask.sum() + 1e-6)


def TV_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def temporal_smooth_loss(x):
    # x: [B, T, HW]
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))


# ============================================================
# Temporal Decay
# ============================================================
class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRUD model.
    Please refer to the original paper :cite:`che2018GRUD` for more deinails.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

# ============================================================
# Mask-Guided Encoder Block
# ============================================================
class MaskGatedConv(nn.Module):
    """
    Input:
        x    : [B, Cin, H, W]
        mask : [B, 1,   H, W]
    Output:
        y    : [B, Cout, H, W]
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Conv2d(1, out_c, 3, 1, 1)

    def forward(self, x, mask):
        feat = self.conv(x)                       # [B, Cout, H, W]
        gate = torch.sigmoid(self.gate(mask))     # [B, Cout, H, W]
        return feat * gate


# ============================================================
# Residual Block
# ============================================================
class ResidualBlock(nn.Module):
    """
    Input / Output:
        x : [B, C, H, W]
    """
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ============================================================
# Mask-Aware Decoder Gate
# ============================================================
class MaskDecoderGate(nn.Module):
    """
    Input:
        feat : [B, C, H, W]
        mask : [B, 1, H, W]
    Output:
        feat : [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat, mask):
        return feat * self.gate(mask)


# ============================================================
# Final Spatial UNet (H × W output)
# ============================================================
class SpatialUNet(nn.Module):
    """
    Input:
        x    : [B, 2, H, W]   (value + mask channel)
        mask : [B, 1, H, W]
    Output:
        out  : [B, 1, H, W]
        conf : [B, 1, H, W]
    """
    def __init__(self, in_channels=2, base=64):
        super().__init__()

        # ---------------- Encoder ----------------
        self.enc1 = MaskGatedConv(in_channels, base)        # [B,64,H,W]
        self.enc2 = MaskGatedConv(base, base * 2)           # [B,128,H/2,W/2]
        self.enc3 = MaskGatedConv(base * 2, base * 4)       # [B,256,H/4,W/4]

        self.pool = nn.MaxPool2d(2)

        # ---------------- Bottleneck ----------------
        self.mid = nn.Sequential(
            ResidualBlock(base * 4),
            ResidualBlock(base * 4)
        )                                                   # [B,256,H/8,W/8]

        # ---------------- Decoder ----------------
        # H/8 -> H/4
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 6, base * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base * 2)
        )
        self.dec3_gate = MaskDecoderGate(base * 2)

        # H/4 -> H/2
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 3, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            ResidualBlock(base)
        )
        self.dec2_gate = MaskDecoderGate(base)

        # H/2 -> H
        self.up1 = nn.ConvTranspose2d(base, base, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            ResidualBlock(base)
        )
        self.dec1_gate = MaskDecoderGate(base)

        # ---------------- Output heads ----------------
        self.value_head = nn.Conv2d(base, 1, 3, 1, 1)
        self.conf_head = nn.Sequential(
            nn.Conv2d(base, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # Multi-scale masks
        mask1 = mask                                  # [B,1,H,W]
        mask2 = self.pool(mask1)                      # [B,1,H/2,W/2]
        mask3 = self.pool(mask2)                      # [B,1,H/4,W/4]

        # Encoder
        e1 = self.enc1(x, mask1)                      # [B,64,H,W]
        e2 = self.enc2(self.pool(e1), mask2)          # [B,128,H/2,W/2]
        e3 = self.enc3(self.pool(e2), mask3)          # [B,256,H/4,W/4]

        # Bottleneck
        m = self.mid(self.pool(e3))                   # [B,256,H/8,W/8]

        # Decoder
        d3 = self.up3(m)                              # [B,128,H/4,W/4]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.dec3_gate(d3, mask3)

        d2 = self.up2(d3)                             # [B,64,H/2,W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2_gate(d2, mask2)

        d1 = self.up1(d2)                             # [B,64,H,W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.dec1_gate(d1, mask1)

        # Output
        value = self.value_head(d1)                   # [B,1,H,W]
        conf  = self.conf_head(d1)                    # [B,1,H,W]
        out = value * conf

        return out


# ============================================================
# ST-UNet-RITS
# ============================================================
class ST_UNet_RITS(nn.Module):
    def __init__(
        self,
        seq_len: int,
        img_size: int,
        rnn_hidden_size: int = 256,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.H = self.W = img_size
        self.device = device
        self.feature_dim = img_size * img_size

        self.rnn_cell = nn.LSTMCell(
            self.feature_dim * 2, rnn_hidden_size
        )

        self.temp_decay = TemporalDecay(
            input_size=self.feature_dim, output_size=rnn_hidden_size, diag=False
        )

        self.hist_reg = nn.Linear(rnn_hidden_size, self.feature_dim)

        self.spatial_unet = SpatialUNet()

    def forward(self, inputs, direction="forward", task="train"):
        X = inputs[direction]["X"]
        X_masked = inputs[direction]["X_mask"]
        M = inputs[direction]["missing_mask"]
        D = inputs[direction]["deltas"]

        B = X.size(0)
        h = torch.zeros(B, self.hist_reg.in_features, device=self.device)
        c = torch.zeros_like(h)

        rec_loss, tv_loss,rec_mae = 0.0, 0.0, 0.0
        estimations = []

        for t in range(self.seq_len):
            x_true = X[:, t]
            x = X_masked[:, t]
            m = M[:, t]
            d = D[:, t]

            h = h * self.temp_decay(d)
            x_hist = self.hist_reg(h)
            x_c = m * x + (1 - m) * x_hist

            spatial_in = torch.cat([
                x_c.view(B, 1, self.H, self.W),
                m.view(B, 1, self.H, self.W)
            ], dim=1)

            x_spatial = self.spatial_unet(spatial_in, m.view(B, 1, self.H, self.W))
            x_spatial = x_spatial.view(B, -1)

            if task != "test":
                rec_loss += cal_mae(x_spatial, x_true, 1 - m) + 0.1 * cal_mae(x_spatial, x_true, m)
                tv_loss += TV_loss(
                    x_spatial.view(B, 1, self.H, self.W) *
                    (1 - m).view(B, 1, self.H, self.W)
                )
            rec_mae += cal_mae(x_spatial, x_true, (1-m))
            c_t = m * x + (1 - m) * x_spatial
            estimations.append(c_t.unsqueeze(1))

            rnn_in = torch.cat([c_t, m], dim=1)
            h, c = self.rnn_cell(rnn_in, (h, c))

        estimations = torch.cat(estimations, dim=1)
        imputed = M * X_masked + (1 - M) * estimations

        temp_loss = temporal_smooth_loss(estimations)
        loss = rec_loss / self.seq_len + 0.05 * tv_loss / self.seq_len + 0.1 * temp_loss
        rec_mae = rec_mae / self.seq_len

        return {
            "loss": loss,
            "imputed_data": imputed,
            "reconstruction_MAE": rec_mae,
            "temporal_loss": temp_loss
        }
