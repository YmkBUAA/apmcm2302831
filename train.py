import torch
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from utils import read_dir, preprocess, contrastive_loss
import random
from model import ResNet_18, Decoder, ResNet_50, ResNet_101
from tqdm import tqdm

# decoder loss
from utils import decoder_loss
# 超参数
'''
neg_per_q = 1  # 负样本个数
t = 0.05  # 温度系数
batchsize = 8
epochs = 11
'''
# 创建模型
# model = ResNet_18(num_classes=2)
# model.train()

# 解码器
# decoder = Decoder(input_dim=2, hidden_size=2, output_dim=1)
# 优化器
# optimizer1 = optim.Adam(params=model.parameters(), lr=1e-5)
# optimizer2 = optim.Adam(params=decoder.parameters(), lr=1e-5)


def train_encoder(pos_imgs, neg_imgs):
    # encoder 参数
    model_name = []
    neg_per_q = 1  # 负样本个数
    batchsize = 8
    epochs = 11
    # 创建模型
    model = ResNet_18(num_classes=2)
    # model = ResNet_50(num_classes=2)
    # model = ResNet_101(num_classes=2)
    model.train()
    optimizer1 = optim.Adam(params=model.parameters(), lr=1e-5)
    num_neg = len(neg_imgs)
    batch_count = 0
    for epoch in range(epochs):
        for idx in tqdm(range(len(pos_imgs))):
            # 读入正样本
            query_batch = preprocess(pos_imgs[idx])
            key_idx = idx
            while key_idx == idx:
                key_idx = random.randint(0, len(pos_imgs)-1)
            if key_idx >= len(pos_imgs):
                print('------------!!!!!!--------')
            key_batch = preprocess(pos_imgs[key_idx])

            # TODO 读入k个负样本 DONE!
            neg_idx = random.sample(range(0, num_neg), neg_per_q)
            neg_batches = []
            for i in neg_idx:
                neg_batches.append(preprocess(neg_imgs[i]))
            neg_batches = torch.vstack(neg_batches)

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                query_batch = query_batch.to('cuda')
                key_batch = key_batch.to('cuda')
                neg_batches = neg_batches.to('cuda')

            if batch_count == 0:
                query = query_batch
                key = key_batch
                negative = neg_batches
                batch_count += 1
            elif batch_count < batchsize:
                query = torch.vstack([query, query_batch])
                key = torch.vstack([key, key_batch])
                negative = torch.vstack([negative, neg_batches])
                batch_count += 1
            else:
                # TODO 计算损失函数 反向传播
                if torch.cuda.is_available():
                    query = query.to('cuda')
                    key = key.to('cuda')
                    negative = negative.to('cuda')
                    model.to('cuda')
                    # decoder.to('cuda')

                model.zero_grad()
                #### decoder ###
                # decoder.zero_grad()
                ################
                q = model(query)
                k = model(key)
                k.detach()  # no gradients to key
                neg = model(negative)

                # q_prob = decoder(q)
                # neg_prob = decoder(neg)
                encoder_loss = contrastive_loss(q, k, neg).mean()
                # decoder_loss = decoder_loss(q_prob, neg_prob).mean()
                # 分开更新参数
                encoder_loss.backward()
                optimizer1.step()
                # decoder_loss.backward()
                # optimizer2.step()
                # 重置
                batch_count = 0
        print(epoch, 'contrastive_loss', encoder_loss)
        if epoch % 2 == 0:
            save_name_en = 'runs\\resnet_101_train_' + str(epoch) + '.pt'
            save_name_de = 'runs\\decoder_' + str(epoch) + '.pt'
            torch.save(model, save_name_en)
            model_name.append(save_name_en)
            # torch.save(decoder, save_name_de)
    return model_name


def train_decoder(pos_imgs, neg_imgs,save_name_encoder):
    # decoder 参数
    neg_per_q = 1  # 负样本个数
    batchsize = 8
    epochs = 5
    model = torch.load(save_name_encoder)
    decoder = Decoder(input_dim=2, hidden_size=2, output_dim=1)
    optimizer2 = optim.Adam(params=decoder.parameters(), lr=1e-5)
    num_neg = len(neg_imgs)
    batch_count = 0
    for epoch in range(epochs):
        for idx in tqdm(range(len(pos_imgs))):
            # 读入正样本
            query_batch = preprocess(pos_imgs[idx])
            key_idx = idx
            while key_idx == idx:
                key_idx = random.randint(0, len(pos_imgs) - 1)
            if key_idx >= len(pos_imgs):
                print('------------!!!!!!--------')
            key_batch = preprocess(pos_imgs[key_idx])

            # TODO 读入k个负样本 DONE!
            neg_idx = random.sample(range(0, num_neg), neg_per_q)
            neg_batches = []
            for i in neg_idx:
                neg_batches.append(preprocess(neg_imgs[i]))
            neg_batches = torch.vstack(neg_batches)

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                query_batch = query_batch.to('cuda')
                key_batch = key_batch.to('cuda')
                neg_batches = neg_batches.to('cuda')

            if batch_count == 0:
                query = query_batch
                key = key_batch
                negative = neg_batches
                batch_count += 1
            elif batch_count < batchsize:
                query = torch.vstack([query, query_batch])
                key = torch.vstack([key, key_batch])
                negative = torch.vstack([negative, neg_batches])
                batch_count += 1
            else:
                # TODO 计算损失函数 反向传播
                if torch.cuda.is_available():
                    query = query.to('cuda')
                    key = key.to('cuda')
                    negative = negative.to('cuda')
                    model.to('cuda')
                    decoder.to('cuda')

                model.zero_grad()
                #### decoder ###
                # decoder.zero_grad()
                ################
                with torch.no_grad():
                    q = model(query)
                    k = model(key)
                    neg = model(negative)

                q_prob = decoder(q)
                neg_prob = decoder(neg)
                # encoder_loss = contrastive_loss(q, k, neg).mean()
                de_loss = decoder_loss(q_prob, neg_prob).mean()
                # 分开更新参数
                # encoder_loss.backward()
                # optimizer1.step()
                de_loss.backward()
                optimizer2.step()
                # 重置
                batch_count = 0
        print(epoch, 'contrastive_loss', de_loss)
        '''
        if epoch % 5 == 0:
            # save_name_en = 'runs\\encoder_' + str(epoch) + '.pt'
            save_name_de = 'runs\\decoder_' + str(epoch) + '.pt'
            # torch.save(model, save_name_en)
            torch.save(decoder, save_name_de)
        '''
        # save_name_en = 'runs\\encoder_' + str(epoch) + '.pt'
        save_name_de = 'runs\\decoder_' + str(epoch) + '.pt'
        # torch.save(model, save_name_en)
        torch.save(decoder, save_name_de)


# 读入数据
trainset, testset = read_dir(r'D:\code\apmcm\Attachment\Attachment\Attachment2')
pos_imgs = []
neg_imgs = []
for key in trainset.keys():
    if key == 'Apple':
        pos_imgs += trainset[key]
    else:
        neg_imgs += trainset[key]
model_name = train_encoder(pos_imgs, neg_imgs)
# model_name = ['runs\\resnet_50_train_2.pt', 'runs\\resnet_50_train_4.pt', 'runs\\resnet_50_train_6.pt',
#                'runs\\resnet_50_train_8.pt','runs\\resnet_50_train_10.pt']
#     , 'runs\\resnet_50_train_12.pt','runs\\1126encoder_train_14.pt','runs\\1126encoder_train_16.pt']
for name in model_name:
    print('--------------')
    print(name)
    model = torch.load(name)
    model.eval()
    for key in testset.keys():
        imgs = testset[key]
        idx = random.sample(range(0, len(imgs)), 16)
        batches = []
        for i in idx:
            batches.append(preprocess(imgs[i]))
        batches = torch.vstack(batches)
        if torch.cuda.is_available():
            batches = batches.to('cuda')
            model.to('cuda')
        output = model(batches)
        expectation = torch.mean(output, dim=0, keepdim=True)
        variation = torch.std(output, dim=0, keepdim=True)
        print(key, expectation, '---', variation)

#  encoder_train_4.pt
#  encoder_train_8.pt
