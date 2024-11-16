import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchinfo import summary

from utils.layers import make_activation_layer


class FeedForward(LightningModule):
    def __init__(
        self,
        layer_sizes,
        activation_type='relu',
        learning_rate=1e-3,
        reg_lambda=1e-3
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        assert len(layer_sizes) >= 2
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(make_activation_layer(activation_type))
        layers.pop()  # no activation after last layer

        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.flatten(x)
        y = self.linears(y)
        y = self.softmax(y)
        return y

    def training_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_act = batch
        y_pred = self(x)

        loss = F.cross_entropy(y_act, y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_lambda
        )


def main():
    model = FeedForward([3072, 100, 10, 4])
    print(model)
    summary(model, input_size=(32, 3, 32, 32), col_names=(
                # "input_size",
                "output_size",
                "kernel_size",
                # "num_params",
                # "mult_adds",
            ))


if __name__ == '__main__':
    main()