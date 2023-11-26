import os

import torch
from PIL import Image
from torchvision import transforms
from utils import preprocess, read_dir
import numpy as np
import matplotlib.pyplot as plt
'''
# encoder = torch.load('runs\\1126encoder_train_6.pt')
encoder = torch.load('runs\\resnet_50_train_8.pt')
encoder.eval()
dirname_input = r'D:\code\apmcm\Attachment\Attachment\Attachment3'
output = np.zeros([1, len(os.listdir(dirname_input))])
print(output.shape)
for index in range(len(os.listdir(dirname_input))):
    imgname = dirname_input+'\\Fruit ('+str(index+1)+').jpg'
    input = preprocess(Image.open(imgname)).to('cuda')
    out = encoder(input)
    print(index+1, out)
    if 0.85 <= out[:, 1] <= 0.95:
        output[:, index] = 1
'''
output = np.load('output.npy').squeeze(0)
print(output.shape[0])
x = np.arange(0, output.shape[0])
print(x.shape)
# plt.hist2d(x, output, bins=5000)
plt.plot(x, output, label='maturity', color='blue', linestyle='-', linewidth=0.5, marker='o', markersize=1)
plt.title("prediction output on attachment3")
plt.xlabel("ID of apple image")
plt.ylabel("prediction")
plt.show()


'''
runs\1126encoder_train_20.pt
Apple tensor([[0.0590, 0.9410]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0157, 0.0157]], device='cuda:0', grad_fn=<StdBackward0>)
Carambola tensor([[0.2343, 0.7657]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0867, 0.0867]], device='cuda:0', grad_fn=<StdBackward0>)
Pear tensor([[0.1019, 0.8981]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0260, 0.0260]], device='cuda:0', grad_fn=<StdBackward0>)
Plum tensor([[0.1270, 0.8730]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0319, 0.0319]], device='cuda:0', grad_fn=<StdBackward0>)
Tomatoes tensor([[0.1271, 0.8729]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.1213, 0.1213]], device='cuda:0', grad_fn=<StdBackward0>)
--------------
runs\1126encoder_train_12.pt
Apple tensor([[0.1167, 0.8833]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0300, 0.0300]], device='cuda:0', grad_fn=<StdBackward0>)
Carambola tensor([[0.2337, 0.7663]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.1626, 0.1626]], device='cuda:0', grad_fn=<StdBackward0>)
Pear tensor([[0.1831, 0.8169]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0331, 0.0331]], device='cuda:0', grad_fn=<StdBackward0>)
Plum tensor([[0.2767, 0.7233]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0773, 0.0773]], device='cuda:0', grad_fn=<StdBackward0>)
Tomatoes tensor([[0.0554, 0.9446]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0611, 0.0611]], device='cuda:0', grad_fn=<StdBackward0>)
--------------
--------------
runs\1126encoder_train_6.pt
Apple tensor([[0.2320, 0.7680]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0926, 0.0926]], device='cuda:0', grad_fn=<StdBackward0>)
Carambola tensor([[0.1011, 0.8989]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.1028, 0.1028]], device='cuda:0', grad_fn=<StdBackward0>)
Pear tensor([[0.3598, 0.6402]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0563, 0.0563]], device='cuda:0', grad_fn=<StdBackward0>)
Plum tensor([[0.3910, 0.6090]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0565, 0.0565]], device='cuda:0', grad_fn=<StdBackward0>)
Tomatoes tensor([[0.0523, 0.9477]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.1044, 0.1044]], device='cuda:0', grad_fn=<StdBackward0>)

--------------
runs\resnet_50_train_8.pt
Apple tensor([[0.2910, 0.7090]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.1687, 0.1687]], device='cuda:0', grad_fn=<StdBackward0>)
Carambola tensor([[0.4664, 0.5336]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.3062, 0.3062]], device='cuda:0', grad_fn=<StdBackward0>)
Pear tensor([[0.1909, 0.8091]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0354, 0.0354]], device='cuda:0', grad_fn=<StdBackward0>)
Plum tensor([[0.1087, 0.8913]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.0707, 0.0707]], device='cuda:0', grad_fn=<StdBackward0>)
Tomatoes tensor([[0.6348, 0.3652]], device='cuda:0', grad_fn=<MeanBackward1>) --- tensor([[0.2279, 0.2279]], device='cuda:0', grad_fn=<StdBackward0>)

'''