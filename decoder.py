import torch
from PIL import Image
from torchvision import transforms
from utils import preprocess, read_dir
import random
import numpy as np
encoder = torch.load(r'runs\1126encoder_train_12.pt')
# encoder2 = torch.load(r'runs\1126encoder_train_6.pt')
encoder.eval()
# encoder2.eval()
trainset, testset = read_dir(r'D:\code\apmcm\Attachment\Attachment\Attachment2')
# model_name = ['runs\\resnet_50_train_2.pt', 'runs\\resnet_50_train_4.pt', 'runs\\resnet_50_train_6.pt',
#                'runs\\resnet_50_train_8.pt','runs\\resnet_50_train_10.pt']
#     , 'runs\\resnet_50_train_12.pt','runs\\1126encoder_train_14.pt','runs\\1126encoder_train_16.pt']

num_right = 0
num_false = 0
for key in testset.keys():
    for img in testset[key]:
        input = preprocess(img)
        if torch.cuda.is_available():
            input = input.to('cuda')
            encoder = encoder.to('cuda')
            # encoder2 = encoder2.to('cuda')
        output = encoder(input)
        if 0.85 <= output[:, 1] <= 0.95:
            if key == 'Apple':
                num_right += 1
            else:
                num_false += 1
        else:
            if key == 'Apple':
                num_false += 1
            else:
                num_right += 1
print('---------')
print(num_right/(num_false+num_right))


#  encoder_train_4.pt
#  encoder_train_8.pt
