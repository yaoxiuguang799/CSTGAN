import torch
import torch.nn as nn


# =========================================================
# 1. Spatial Discriminator
# =========================================================
class SpatialDiscriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            block(2, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            nn.Conv2d(base * 4, 1, 4, 1, 1)
        )

    def forward(self, x, mask):
        x = x * (1 - mask)
        inp = torch.cat([x, 1 - mask], dim=1)
        return self.net(inp)
    
# =========================================================
# 2. Temporal Discriminator
# =========================================================
class TemporalDiscriminator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        # x, mask: [B, T, HW]
        x = x * (1 - mask)

        B, T, HW = x.shape
        x = x.view(B * HW, T, 1)
        m = (1 - mask).view(B * HW, T, 1)

        seq = torch.cat([x, m], dim=-1)
        _, h = self.gru(seq)
        out = self.fc(h[-1])

        return out.view(B, HW).mean(dim=1, keepdim=True)


# =========================================================
# 3. WGAN-GP 梯度惩罚
# =========================================================
def gradient_penalty(D, real, fake, mask, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)

    out = D(interp, mask)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

