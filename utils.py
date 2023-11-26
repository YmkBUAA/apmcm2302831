from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torchvision import transforms

'''
# 图像读取，分别读取正类与负类
def read_dir(directory):
    pos_imgs=[]
    neg_imgs=[]
    for fruit in os.listdir(directory): # read attachment2/apple  etc
        path = directory + '\\' + fruit
        for filename in os.listdir(path):
            filename = path + '\\' + filename
            img = Image.open(filename)
            img_temp = img.copy()
            if fruit == 'Apple':
                pos_imgs.append(img_temp)
            else:
                neg_imgs.append(img_temp)
            img.close()
    return pos_imgs, neg_imgs
'''
def read_dir(directory):
    # imgs = {}
    train_set = {}
    test_set = {}
    for fruit in os.listdir(directory): # read attachment2/apple  etc
        # imgs[fruit] = []
        train_set[fruit] = []
        test_set[fruit] = []
        path = directory + '\\' + fruit
        # 训练集比例:0.9
        for i, filename in enumerate(os.listdir(path)):
            filename = path + '\\' + filename
            img = Image.open(filename)
            img_temp = img.copy()
            # imgs[fruit].append(img_temp)
            img.close()
            # if random.uniform(0, 1) <= 0.9:
            if i/len(os.listdir(path)) <= 0.9:
                train_set[fruit].append(img_temp)
            else:
                test_set[fruit].append(img_temp)
    return train_set, test_set


def preprocess(input_image):
    # input_batch: 1x3x224x224
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = trans(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch

def contrastive_loss(q, k, neg):
    '''
    Calculating loss of encoder
    Args:
        q: query [batch, encoder_output_dim]
        k: key [batch, encoder_output_dim]
        neg: negative [batch,n, encoder_output_dim]

    Returns:
        loss: contrastive loss
    '''
    temp = 0.7
    b = q.shape[0]
    neg = neg.unsqueeze(0).view(b, -1, 2)
    n = neg.shape[1]
    l_pos = torch.bmm(q.view(b, 1, -1), k.view(b, -1, 1))  # (b,1,1)
    l_neg = torch.bmm(q.view(b, 1, -1), neg.transpose(1, 2))  # (b,1,N)

    logits = torch.cat([l_pos.view(b, 1), l_neg.view(b, n)], dim=1).to('cuda')

    labels = torch.zeros(b, dtype=torch.long).to('cuda')  # label直接就是0，表示第0个是真值
    # labels = labels.to(ptu.device)
    cross_entropy_loss = nn.CrossEntropyLoss().to('cuda')
    loss = cross_entropy_loss(logits / temp, labels)
    # print(logits, labels, loss)
    return loss


def decoder_loss(q_prob, neg_prob):
    loss = -torch.log(q_prob)+torch.log(torch.ones_like(neg_prob)-neg_prob)
    return loss


def test():
    input_image = Image.open(r'/ultralytics/ultralytics/1.jpg')
    input_tensor = preprocess(input_image)
    # input_tensor = torch.stack([input_tensor, input_tensor], dim=1)
    input_tensor = torch.vstack([input_tensor, input_tensor])
    input_tensor = input_tensor.view(2, -1)
    loss = contrastive_loss(input_tensor, input_tensor, input_tensor)
    print(input_tensor.shape)
    print(loss)
