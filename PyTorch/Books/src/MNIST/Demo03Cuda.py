import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

data_train = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
data_test = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=256,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=256,
                                               shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x

# 实例化
model = Model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 定义损失函数对象
cost = torch.nn.CrossEntropyLoss()
# 定义优化函数
optimizer = torch.optim.Adam(model.parameters())

# 为了节约时间, 这里只训练一个epoch
n_epoch = 1

for epoch in range(n_epoch):
    running_loss = 0.0
    running_correct = 0
    print("Epoch:{}/{}".format(epoch, n_epoch))
    print("-"*20)

    # 训练部分
    for data in data_loader_train:
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)
        x_train = x_train.to(device)
        y_train = y_train.to(device)


        outputs = model(x_train)  # 前向传播
        pred = torch.max(outputs.data, 1)[1]

        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        # loss = loss.cuda()

        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)

    # 验证部分
    testing_correct = 0
    for data in data_loader_test:
        x_test, y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        pred = torch.max(outputs.data, 1)[1]
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                      100*running_correct / len(data_train),
                                                                                      100*testing_correct / len(data_test)))

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 4,
                                               shuffle = True)
x_test, y_test = next(iter(data_loader_test))
inputs = Variable(x_test)
inputs = inputs.to(device)
pred = model(inputs)
pred = torch.max(pred, 1)[1]

print("Predict Label is:", [i for i in pred.data])
print("Real Label is:", [i for i in y_test])

img = torchvision.utils.make_grid(x_test)
img = img.numpy().transpose(1, 2, 0)

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]

img = img * std + mean

plt.imshow(img)
plt.show()