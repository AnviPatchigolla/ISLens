import torch
import torch.nn as nn

from features import FRAME_DIM


class ISLensLSTM(nn.Module):
    def __init__(self, input_size=FRAME_DIM, hidden_size=64, num_layers=1, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))


def build_model(classes, hidden_size=64, num_layers=1):
    return ISLensLSTM(
        input_size=FRAME_DIM,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(classes),
    )


def save_checkpoint(path, model, classes, hidden_size, num_layers, seq_len):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "input_size": FRAME_DIM,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model = ISLensLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=len(ckpt["classes"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
