from torchvision import transforms
from PIL import Image
import os
import torch
from torch import nn
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification.f_beta import F1Score
import torch.nn.functional as F
import json
import shutil
from torchinfo import summary

# pred = torch.FloatTensor([
#     [-10, -2, -3],
#     [10, -2, -3],
#     [-10, -2, 3],
# ])
# labels = torch.LongTensor([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0, 1],
# ])
# print(F.cross_entropy(pred, labels))

# metrics = MetricCollection([
#     Accuracy(),
#     F1Score(num_classes=3)
# ])

# print(metrics(pred, labels.argmax(dim=-1)))

# d = {
#     'ami': torch.tensor(1.65),
#     'tumi': torch.tensor(2.3)
# }
# for key in d.keys():
#     d[key] = float(d[key])
# print(json.dumps(d, indent=4))

# from datamodules.simple_datamodule import get_data

# x, y = get_data('train')
# print(len(x), len(y))

# import torchvision.models as models

# model = models.inception_v3(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# print(model)
# summary(model, input_size=(32, 3, 299, 299), col_names=(
#             "input_size",
#             "output_size",
#             "kernel_size",
#             # "num_params",
#             # "mult_adds",
#         ))

# y = model(torch.ones(32, 3, 299, 299))
# print(y.logits)

from pytorch_lightning import Trainer
from models.transferer import Transferer
from datamodules.simple_datamodule import SimpleDataModule

model = Transferer.load_from_checkpoint('saved_models/model_v25.ckpt')

dm = SimpleDataModule(batch_size=64)

trainer = Trainer(
    # max_epochs=params['max_epochs'],
    # gpus=-1,
    # benchmark=True,
)

metrics = trainer.test(model, dm)
print(metrics)
