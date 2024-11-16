import os
import random
from PIL import Image
import numpy as np
from torchvision import transforms


dirs = [
    'chest_xray/train',
    'chest_xray/val',
    'chest_xray/test',
]

# for dir in dirs:
#     from_dir = f'{dir}/PNEUMONIA'
#     bac_dir = f'{dir}/bacteria'
#     vir_dir = f'{dir}/virus'

#     if not os.path.isdir(bac_dir):
#         os.makedirs(bac_dir)
#     if not os.path.isdir(vir_dir):
#         os.makedirs(vir_dir)

#     for img_name in os.listdir(from_dir):
#         img_path = f'{from_dir}/{img_name}'
#         img = Image.open(img_path)
#         if 'bacteria' in img_name:
#             img.save(f'{bac_dir}/{img_name}')
#         elif 'virus' in img_name:
#             img.save(f'{vir_dir}/{img_name}')
#         else:
#             print(img_path)
        

for cls in ('normal', 'bacteria', 'virus'):
    all = []
    for dir in dirs:
        mdir = f'{dir}/{cls}'
        all += [(f'{mdir}/{img_name}', img_name) for img_name in os.listdir(mdir)]

    l = len(all)
    a, b = int(l*.8), int(l*.9)
    random.shuffle(all)
    train, val, test = all[:a], all[a:b], all[b:]
    print(len(train), len(val), len(test))

    ddir = 'xray-300-420'
    h, w = 300, 420
    new_dir = f'{ddir}/train/{cls}'
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    for img_path, img_name in train:
        img = Image.open(img_path)
        resize = transforms.Resize((h, w))
        img = resize(img)
        img.save(f'{new_dir}/{img_name}')

    new_dir = f'{ddir}/val/{cls}'
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    for img_path, img_name in val:
        img = Image.open(img_path)
        resize = transforms.Resize((h, w))
        img = resize(img)
        img.save(f'{new_dir}/{img_name}')

    new_dir = f'{ddir}/test/{cls}'
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    for img_path, img_name in test:
        img = Image.open(img_path)
        resize = transforms.Resize((h, w))
        img = resize(img)
        img.save(f'{new_dir}/{img_name}')
    
