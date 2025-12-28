import torch
from torch import nn


class TCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int, horizon: int) -> None:
        super().__init__()
        layers = []
        in_channels = input_channels
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        features = self.tcn(x)
        # 取最后一个时间步作为序列表示
        last = features[:, :, -1]
        return self.head(last)
