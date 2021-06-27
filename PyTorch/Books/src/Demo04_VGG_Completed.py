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

data_dir = "DogsVSCats"
data_transform = {x: transforms.Compose([transforms.Resize([128, 128]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                  for x in ["train", "valid"]}

image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=16,
                                             shuffle=True)
              for x in ["train", "valid"]}


# 实例化
model = models.vgg16(pretrained=True)

# 修改模型
for parameter in model.parameters():
    parameter.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2)
                                       )


# 使用GPU加速
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model.cuda()

summary(model, (3, 224, 224))

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch: {} / {}".format(epoch, epoch_n - 1))
    print("-"*50)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)

        else:
            print("Validating...")
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            x, y = data

            if Use_gpu:
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)

            y_pred = model(x)

            pred = torch.max(y_pred.data, 1)[1]

            optimizer.zero_grad()

            loss = loss_f(y_pred, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f}%, Train Acc:{:.4f}%".format(batch, running_loss / batch, 100 * running_corrects / (16 * batch)))

        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])

        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))
        time_end = time.time() - time_open
        print("The All Time Is:", time_end)

