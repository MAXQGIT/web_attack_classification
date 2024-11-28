import torch.nn as nn
from Config import Config


class LSTM_MODEL(nn.Module):
    def __init__(self, cfg):
        super(LSTM_MODEL, self).__init__()
        self.lstm = nn.LSTM(cfg.input_dim, cfg.hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(cfg.hidden, num_heads=8)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.linear = nn.Linear(cfg.hidden, cfg.output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        # y, attn_output_weights = self.attention(x, x, x)
        # y = x + y
        x = self.relu(x)
        y, _ = self.gru(x)
        # y = x + y
        y = self.linear(y)
        return y


if __name__ == '__main__':
    cfg = Config()
    model = LSTM_MODEL(cfg)
    total_num = sum(pre.numel() for pre in model.parameters())
    print('模型总参数：{}'.format(total_num))
