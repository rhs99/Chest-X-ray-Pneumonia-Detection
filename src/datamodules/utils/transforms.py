from torchvision import transforms
from PIL import Image
import torch


def load_image_and_transform(img_path):
    img = Image.open(img_path)
    tense = transforms.ToTensor()
    x = tense(img)
    x = torch.concat((x, x, x))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = norm(x)
    # print(x.shape)
    return x


# load_image_and_transform('xray_300_420/train/normal/IM-0003-0001.jpeg')
