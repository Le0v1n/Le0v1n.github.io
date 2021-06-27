import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
from tqdm import tqdm
import argparse
import numpy
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=16, help="批大小")
parser.add_argument("--epochs", type=int, default=10, help="训练总数")
parser.add_argument('--make-dir', type=bool, default=False, help="是否需要创建train, valid, test文件夹")
parser.add_argument('--device', type=str, default='gpu', help="cpu or cuda")
parser.add_argument('--valid', type=bool, default=True, help='训练过程中验证模型')
parser.add_argument('--save-dir', type=str, default='weights', help='模型存储位置')
parser.add_argument('--model', type=str, default='resnet',choices=['resnet', 'vgg16'], help='模型保存名称')
parser.add_argument('--resume', type=str, default='weights/resnet50_70', help='恢复模型')
parser.add_argument('--save-loss', type=str, default='loss.txt', help='保存损失')
parser.add_argument('--works', type=int, default=4, help='线程数')

opt = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[.5],
                                                     std=[.5])])

dataset_train = datasets.MNIST(root="../MNIST//data",
                               transform=transform,
                               train=True,
                               download=True)

dataset_test = datasets.MNIST(root="../MNIST//data",
                              transform=transform,
                              train=False)

train_load = torch.utils.data.DataLoader(dataset=dataset_train,
                                         batch_size=opt.batch_size,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=dataset_test,
                                        batch_size=64,
                                        shuffle=True)

images, label = next(iter(train_load))

images_example = torchvision.utils.make_grid(images)
# print(images_example.shape)  # torch.Size([3, 242, 242])

# 由原本的(Channel, Length, Weight)变为(Length, Weight, Channel)这样可以符合正常图片顺序的shape
images_example = images_example.numpy().transpose(1, 2, 0)

mean = [.5, .5, .5]
std = [.5, .5, .5]

images_example = images_example * std + mean

plt.imshow(images_example)
plt.show()


class RNN(torch.nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=28,
                                hidden_size=128,
                                num_layers=1,
                                batch_first=True)
        self.output = torch.nn.Linear(128, 10)

    def forward(self, input):
        output = self.rnn(input, None)[0]
        output = self.output(output[:, -1, :])
        return output


# 搭建模型
model = RNN()
if torch.cuda.is_available():
    model.cuda()
# print(model)

# 定义优化函数
optimizer = torch.optim.Adam(model.parameters())

# 定义损失函数
loss_Fn = torch.nn.CrossEntropyLoss()

for epoch in range(opt.epochs):
    aaaaa = 0.0
    running_correct = 0
    testing_correct = 0
    print("Epoch {}/{}".format(epoch, opt.epochs - 1))
    print("-" * 100)

    for data in tqdm(train_load):
        x_train, y_train = data
        # print(x_train.shape)  # torch.Size([64, 1, 28, 28]) -> 64是batch_size; 1是Channel
        x_train = x_train.view(-1, 28, 28)
        # print(x_train.shape)  # torch.Size([64, 28, 28])
        if torch.cuda.is_available():
            x_train, y_train = Variable(x_train.cuda()), Variable(y_train.cuda())
        else:
            x_train, y_train = Variable(x_train), Variable(y_train)

        # 前向传播
        y_pred = model(x_train)

        # 计算损失
        loss = loss_Fn(y_pred, y_train)

        # 预测结果为pred
        pred = torch.max(y_pred.data, 1)[1]

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 梯度更新
        optimizer.step()

        aaaaa += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    for data in tqdm(test_load):
        x_test, y_test = data
        x_test = x_test.view(-1, 28, 28)
        x_test, y_test = Variable(x_test.cuda()), Variable(y_test.cuda())

        outputs = model(x_test)

        pred = torch.max(outputs.data, 1)[1]
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is {:.4f}%".
          format(aaaaa / len(dataset_train),
                 100 * running_correct / len(dataset_train),
                 100 * testing_correct / len(dataset_test)))

# 测试代码
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                               batch_size=64,
                                               shuffle=True)
x_test, y_test = next(iter(data_loader_test))
x_pred = x_test.view(-1, 28, 28)  # # torch.Size([64, 28, 28])
inputs = Variable(x_pred.cuda())
pred = model(inputs)
pred = torch.max(pred, 1)[1]

print("-" * 60)
print("Prediction Label is:\n{}".format([i.item() for i in pred.data]))
print("-" * 60)
print("Real Label is: \n{}".format([i.item() for i in y_test]))

img = torchvision.utils.make_grid(x_test)
img = img.numpy().transpose(1, 2, 0)

std = [.5, .5, .5]
mean = [.5, .5, .5]

img = img * std + mean
plt.imshow(img)
plt.show()
