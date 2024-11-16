from pytorch_lightning import Trainer
from torchinfo import summary
import json
import csv
import os
import shutil

# from models.sequential_cnn import SequentialCNN
from models.transferer import Transferer
from datamodules.simple_datamodule import SimpleDataModule


with open('parameters.json') as f:
    params = json.load(f)

c, h, w = 3, 299, 299

# model = SequentialCNN(
#     architecture_file=params['archi_file'],
#     input_shape=(c, w, h),
#     learning_rate=params['learning_rate'],
#     reg_lambda=params['lambda']
# )
# model = Transferer(
#     learning_rate=params['learning_rate'],
#     reg_lambda=params['lambda']
# )
model = Transferer.load_from_checkpoint('saved_models/model_v55.ckpt')
print(model)
summary(
    model,
    input_size=(params['batch_size'], c, w, h),
    col_names=("input_size", "output_size", "kernel_size",)
)

dm = SimpleDataModule(batch_size=params['batch_size'])

trainer = Trainer(
    max_epochs=params['max_epochs'],
    gpus=-1,
    benchmark=True,
)

trainer.fit(model, dm)

metrics = trainer.callback_metrics
for key in metrics.keys():
    metrics[key] = f'{float(metrics[key]):.5f}'
print(json.dumps(metrics, indent=4))

version = trainer.logger.version

model_dir = 'saved_models'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model_save_file = os.path.join(model_dir, f'model_v{version}.ckpt')
trainer.save_checkpoint(model_save_file)

archi_dir = 'archs'
if not os.path.isdir(archi_dir):
    os.makedirs(archi_dir)
archi_save_file = os.path.join(archi_dir, f'arch_v{version}.py')
# with open(archi_save_file, 'w') as f:
#     print(model, file=f)
shutil.copy('src/models/transferer.py', archi_save_file)

del params['archi_file']

data_size = len(dm.train_data)

log = {
    'version': version,
    'data_size': data_size,
    'h': h,
    'w': w,
    **params,
    **metrics
}
log_file = 'logs.csv'

if not os.path.isfile(log_file):
    with open(log_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=log.keys())
        writer.writeheader()

with open(log_file, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=log.keys())
    writer.writerow(log)
