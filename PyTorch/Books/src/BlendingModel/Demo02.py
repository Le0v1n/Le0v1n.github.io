import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""加载数据集"""
data_dir = "../DogsVSCats"
data_transform = {x: transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
                  for x in ["train", "valid"]}

image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=16,
                                             shuffle=True)
              for x in ["train", "valid"]}

# x_example, y_example = next(iter(dataloader["train"]))
# example_classes = image_datasets["train"].classes
# index_classes = image_datasets["train"].class_to_idx
# print("x_example:", x_example)
# print("y_example:", y_example)  # tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
# print("example_classes:", example_classes)  # ['cat', 'dog']
# print("index_classes:", index_classes)  # {'cat': 0, 'dog': 1}

"""导入模型"""
model_1 = models.vgg16(pretrained=True)
model_2 = models.resnet50(pretrained=True)

"""修改模型的全连接层"""
for param in model_1.parameters():
    param.requires_grad = False

model_1.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(4096, 4096),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(4096, 2)
                                         )

for param in model_2.parameters():
    param.requires_grad = False

model_2.fc = torch.nn.Linear(2048, 2)

if torch.cuda.is_available():
    model_1.cuda()
    model_2.cuda()

# 定义损失函数
loss_Fn_1 = torch.nn.CrossEntropyLoss()
loss_Fn_2 = torch.nn.CrossEntropyLoss()

# 定义优化函数
optimizer_1 = torch.optim.Adam(model_1.classifier.parameters(), lr=0.00001)
optimizer_2 = torch.optim.Adam(model_2.fc.parameters(), lr=0.00001)

# 定义模型融合权重
weight_1 = 0.6
weight_2 = 1 - weight_1

# 定义训练论述
epoch_n = 5

# 训练模型
time_open = time.time()

for epoch in range(epoch_n):
    print('Epoch{}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)

for phase in ['train', 'valid']:
    if phase == 'train':
        print('Training...')
        model_1.train(True)
        model_2.train(True)
    else:
        print('Validing...')
        model_1.train(False)
        model_2.train(False)

    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_corrects_1 = 0.0
    running_corrects_2 = 0.0
    blending_running_corrects = 0.0

    for batch, data in enumerate(dataloader[phase], 1):
        x, Y = data
        if torch.cuda.is_available():
            x, Y = Variable(x.cuda()), Variable(Y.cuda())
        else:
            x, Y = Variable(x), Variable(Y)

        y_pred_1 = model_1(x)
        y_pred_2 = model_2(x)
        blending_y_pred = y_pred_1 * weight_1 + y_pred_2 * weight_2

        _, pred_1 = torch.max(y_pred_1.data, 1)  # 找出每一行最大值对应的索引值
        _, pred_2 = torch.max(y_pred_2.data, 1)
        _, blending_y_pred = torch.max(blending_y_pred.data, 1)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        loss_1 = loss_Fn_1(y_pred_1, Y)
        loss_2 = loss_Fn_2(y_pred_2, Y)

        if phase == 'train':
            loss_1.backward()
            loss_2.backward()
            optimizer_1.step()
            optimizer_2.step()

        running_loss_1 += loss_1.data.item()
        running_loss_2 += loss_1.data.item()
        running_corrects_1 += torch.sum(pred_1 == Y.data)
        running_corrects_2 += torch.sum(pred_2 == Y.data)
        blending_running_corrects += torch.sum(blending_y_pred == Y.data)

        if batch % 500 == 0 and phase == 'train':
            print(
                'Batch {},Model1 Train Loss:{:.4f},Model1 Train ACC:{:.4f},Model2 Train Loss:{:.4f},Model2 Train ACC:{:.4f},Blending_Model ACC:{:.4f}'
                    .format(batch, running_loss_1 / batch, 100 * running_corrects_1 / (16 * batch),
                            running_loss_2 / batch,
                            100 * running_corrects_2 / (16 * batch), 100 * blending_running_corrects / (16 * batch)))

    epoch_loss_1 = running_loss_1 * 16 / len(image_datasets[phase])
    epoch_acc_1 = 100 * running_corrects_1 / len(image_datasets[phase])
    epoch_loss_2 = running_loss_2 * 16 / len(image_datasets[phase])
    epoch_acc_2 = 100 * running_corrects_2 / len(image_datasets[phase])
    epoch_blending_acc = 100 * blending_running_corrects / len(image_datasets[phase])
    print('Epoch, Model1 Loss:{:.4f},Model1 ACC:{:.4f}%,Model2 Loss:{:.4f},Model2 ACC:{:.4f}%,Blending_Model ACC:{:.4f}'
          .format(epoch_loss_1, epoch_acc_1, epoch_loss_2, epoch_acc_2, epoch_blending_acc))

    time_end = time.time() - time_open
    print(time_end)
