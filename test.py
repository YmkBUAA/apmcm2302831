import torch
from PIL import Image
from torchvision import transforms
from utils import preprocess, read_dir
import random
import os

num = 32


def read_dir2(directory):
    imgs = []
    for filename in os.listdir(directory):  # read attachment2/apple  etc
            filename = directory + '\\' + filename
            img = Image.open(filename)
            img_temp = img.copy()
            imgs.append(img_temp)
            img.close()
    return imgs
model = torch.load('runs\\5.pt')
model.eval()
filename_apple = r'D:\code\apmcm\Attachment\Attachment\Attachment2\Apple'
filename_carambola = r'D:\code\apmcm\Attachment\Attachment\Attachment2\Carambola'
filename_pear = r'D:\code\apmcm\Attachment\Attachment\Attachment2\Pear'
filename_plum = r'D:\code\apmcm\Attachment\Attachment\Attachment2\Plum'
filename_tomatoes = r'D:\code\apmcm\Attachment\Attachment\Attachment2\Tomatoes'


for filename in [filename_apple, filename_carambola, filename_pear, filename_plum, filename_tomatoes]:
    imgs = read_dir2(filename)
    idx = random.sample(range(0, len(imgs)), num)
    batches = []
    for i in idx:
        batches.append(preprocess(imgs[i]))
    batches = torch.vstack(batches)
    if torch.cuda.is_available():
        batches = batches.to('cuda')
        model.to('cuda')
    output = model(batches)
    expectation = torch.mean(output, dim=0, keepdim=True)
    variation = torch.var(output, dim=0, keepdim=True)
    print(filename, expectation, '---', variation)
