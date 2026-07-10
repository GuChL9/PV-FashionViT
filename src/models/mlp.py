from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, img_size: int = 56, in_channels: int = 1, hidden_dim: int = 512,
                 num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

