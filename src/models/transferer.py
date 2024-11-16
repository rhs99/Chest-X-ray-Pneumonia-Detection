import torch
from torch import nn
import pytorch_lightning as pl
from torchinfo import summary
import torchmetrics as tm
import torchvision.models as models
from torchvision.models.inception import InceptionOutputs

from .utils.layers import make_layer


class Transferer(pl.LightningModule):
    def __init__(
        self,
        architecture_file='architecture.txt',
        input_shape=(1, 1000, 700),
        learning_rate=1e-3,
        reg_lambda=1e-3
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        show_metrics = {
            'F1': tm.F1Score(num_classes=3),
        }
        hide_metrics = {
            'Acc': tm.Accuracy(num_classes=3),
            'Precision': tm.Precision(num_classes=3),
            'Recall': tm.Recall(num_classes=3),
            # 'CnfMat': tm.ConfusionMatrix(num_classes=3)  # can't log
        }
        self.train_shows = tm.MetricCollection(show_metrics, prefix='train')
        self.val_shows = tm.MetricCollection(show_metrics, prefix='val')
        self.test_shows = tm.MetricCollection(show_metrics, prefix='test')
        self.train_hides = tm.MetricCollection(hide_metrics, prefix='train')
        self.val_hides = tm.MetricCollection(hide_metrics, prefix='val')
        self.test_hides = tm.MetricCollection(hide_metrics, prefix='test')

    def forward(self, x):
        y = self.model(x)
        return y

    def common_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)

        if isinstance(logits, InceptionOutputs):
            logits = logits.logits

        loss = self.cross_entropy_loss(logits, y)
        self.log(f'{stage}Loss', loss, prog_bar=False, on_epoch=True, on_step=False)

        labels = y.argmax(dim=-1)
        sm = getattr(self, f'{stage}_shows')(logits, labels)
        self.log_dict(sm, prog_bar=True, on_epoch=True, on_step=False)
        hm = getattr(self, f'{stage}_hides')(logits, labels)
        self.log_dict(hm, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.fc.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_lambda
        )


def main():
    model = Transferer(input_shape=(3, 32, 32))
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
