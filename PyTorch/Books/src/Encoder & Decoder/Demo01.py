import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练总数")
    parser.add_argument('--make-dir', type=bool, default=False, help="是否需要创建train, valid, test文件夹")
    parser.add_argument('--device', type=str, default='gpu', help="cpu or cuda")
    parser.add_argument('--valid', type=bool, default=True, help='训练过程中验证模型')
    parser.add_argument('--save-dir', type=str, default='weights', help='模型存储位置')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vgg16'], help='模型保存名称')
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

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=opt.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    images, label = next(iter(train_loader))
    # print(images.shape)  # torch.Size([4, 1, 28, 28])

    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1, 2, 0)
    mean = [.5, .5, .5]
    std = [.5, .5, .5]
    images_example = images_example * std + mean
    plt.imshow(images_example)
    plt.show()

    # 加入Mosaic
    # print(type(images_example))  # <class 'numpy.ndarray'>
    # print(images_example.shape)  # (32, 122, 3)
    # print(type(*images_example))  # TypeError: type() takes 1 or 3 arguments
    # print(*images_example.shape)
    noisy_images = images_example + 0.5 * np.random.randn(*images_example.shape)

    # print(type(noisy_images))  # <class 'numpy.ndarray'>
    noisy_images = np.clip(noisy_images, 0., 1.)
    plt.imshow(noisy_images)
    plt.show()

    # 模型实例化
    model = AutoEncoder()
    # print(model)
    if opt.device:
        model.cuda()

    # 训练部分
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(opt.epochs):
        running_loss = 0.0

        print("Epoch {}/{}".format(epoch + 1, opt.epochs))
        print("-" * 60)

        for data in tqdm(train_loader):
            x_train, _ = data

            noisy_x_train = x_train + 0.5 * torch.randn(*x_train.shape)
            noisy_x_train = torch.clamp(noisy_x_train, 0.1, 1.0)

            if opt.device:
                x_train, noisy_x_train = Variable(x_train.view(-1, 28 * 28).cuda()), Variable(
                    noisy_x_train.view(-1, 28 * 28).cuda())
            else:
                x_train, noisy_x_train = Variable(x_train.view(-1, 28 * 28)), Variable(noisy_x_train.view(-1, 28 * 28))

            image_pred = model(noisy_x_train)
            loss = loss_fn(image_pred, x_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Loss is: {:.4f}".format(running_loss / len(dataset_train)))

    # 测试部分
    x_test, _ = next(iter(tqdm(test_loader)))

    img_1 = torchvision.utils.make_grid(x_test)
    img_1 = img_1.numpy().transpose(1, 2, 0)
    std = [.5, .5, .5]
    mean = [.5, .5, .5]
    img_1 = img_1 * std + mean

    plt.figure()
    plt.title("Origin Image")
    plt.imshow(img_1)
    plt.show()

    # 打码
    noisy_x_test = img_1 + .5 * np.random.randn(*img_1.shape)
    noisy_x_test = np.clip(noisy_x_test, 0.0, 1.0)

    plt.figure()
    plt.title("Mosaic Image")
    plt.imshow(noisy_x_test)
    plt.show()

    img_2 = x_test + .5 * torch.randn(*x_test.shape)
    img_2 = torch.clamp(img_2, 0.0, 1.0)

    if opt.device:
        img_2 = Variable(img_2.view(-1, 28 * 28).cuda())
    else:
        img_2 = Variable(img_2.view(-1, 28 * 28))

    test_pred = model(img_2)

    img_test = test_pred.data.view(-1, 1, 28, 28)
    print("type(img_test)", type(img_test))  # <class 'torch.Tensor'>

    img_2 = torchvision.utils.make_grid(img_test)
    print("type(img_2)", type(img_2))  # <class 'torch.Tensor'>
    img_2 = img_2.cpu().numpy().transpose(1, 2, 0)
    img_2 = img_2 * std + mean
    img_2 = np.clip(img_2, 0.0, 1.0)
    plt.figure()
    plt.title("Fixed Image")
    plt.imshow(img_2)
    plt.show()


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 64),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(64, 32),
                                           torch.nn.ReLU())

        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 64),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(64, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 28 * 28))

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output


if __name__ == "__main__":
    main()
