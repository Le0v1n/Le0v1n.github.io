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

# print("dataloader['train']:", dataloader["train"])  # <torch.utils.data.dataloader.DataLoader object at 0x000002379E311D60>
# iteration = iter(dataloader["train"])
# print("iteration:", iteration)
# a, b = next(iteration)
# print("type(a):", type(a))  # <class 'torch.Tensor'>
# print("type(b):", type(b))  # <class 'torch.Tensor'>
# # print(a)  a为训练数据
# print(b)  # tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]) -> b为标签(One-Hot)形式

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

# 开始训练
time_start = time.time()
for epoch in range(epoch_n):
    print("Epoch: {} / {}".format(epoch, epoch_n - 1))
    print("-" * 50)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model_1.train(True)
            model_2.train(True)
        else:
            print("Validating...")
            model_1.train(False)
            model_2.train(False)

        running_loss_1 = 0.0
        running_corrects_1 = 0
        running_loss_2 = 0.0
        running_corrects_2 = 0
        blending_running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            x, y = data
            if torch.cuda.is_available():
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)

            # 前向传播
            y_pred_1 = model_1(x)
            y_pred_2 = model_2(x)
            blending_y_pred = y_pred_1 * weight_1 + y_pred_2 * weight_2

            pred_1 = torch.max(y_pred_1.data, 1)[1]
            pred_2 = torch.max(y_pred_2.data, 1)[1]
            blending_pred = torch.max(blending_y_pred.data, 1)[1]

            # 梯度清0
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # 计算损失
            loss_1 = loss_Fn_1(y_pred_1, y)
            loss_2 = loss_Fn_2(y_pred_2, y)

            """
            先判断是在训练还是在验证: 
                如果在训练则开始进行计算反向传播, 并更新梯度
                如果在验证则开始不进行计算反向传播, 不更新梯度
            """
            if phase == "train":
                # 反向传播
                loss_1.backward()
                loss_2.backward()

                # 梯度更新
                optimizer_1.step()
                optimizer_2.step()

            running_loss_1 += loss_1.item()
            running_corrects_1 += torch.sum(pred_1 == y.data)
            running_loss_2 += loss_2.item()
            running_corrects_2 += torch.sum(pred_2 == y.data)
            blending_running_corrects += torch.sum(blending_pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print("Batch {}:\n "
                      "--------------------------------------------------------------------\n"
                      "Model_1 Train Loss:{:.4f}, "
                      "Model_1 Train Acc:{:.4f}\n"
                      "Model_2 Train Loss:{:.4f}, "
                      "Model_2 Train Acc:{:.4f}\n "
                      "--------------------------------------------------------------------\n"
                      "Blending_Model Acc:{:.4f}%".format(batch,
                                                         running_loss_1 / batch,
                                                         100 * running_corrects_1 / (16 * batch),
                                                         running_loss_2 / batch,
                                                         100 * running_corrects_2 / (16 * batch),
                                                         100 * blending_running_corrects / (16 * batch)
                                                         ))

        # 大的统计(上面那个是每500个batch后的一个小统计)(而且根据代码的设计, 如果是验证, 这也是验证的统计)
        epoch_loss_1 = running_loss_1 * 16 / len(image_datasets[phase])
        epoch_acc_1 = 100 * running_corrects_1 / len(image_datasets[phase])
        epoch_loss_2 = running_loss_2 * 16 / len(image_datasets[phase])
        epoch_acc_2 = 100 * running_corrects_2 / len(image_datasets[phase])
        epoch_blending_acc = 100 * blending_running_corrects / len(image_datasets[phase])

        print("Model_1 Loss:{:.4f}, Model_1 Acc:{:.4f}%\n "
              "Model_2 Loss:{:.4f}, Model_2 Acc:{:.4f}%\n "
              "Blending_Model Acc:{:.4f}%".format(epoch_loss_1,
                                                  epoch_acc_1,
                                                  epoch_loss_2,
                                                  epoch_acc_2,
                                                  epoch_blending_acc
                                                  ))
        # 统计时间
        time_end = time.time()
        print("Total Time is:{}".format(time_end - time_start))
        print("-"*50)
