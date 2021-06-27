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

    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
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

    # 导入数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[.5],
                                                         std=[.5])])
    dataset_train = datasets.MNIST(root="../MNIST//data",
                                   train=True,
                                   transform=transform,
                                   download=True)
    dataset_test = datasets.MNIST(root="../MNIST//data",
                                  train=False,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=opt.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                              batch_size=opt.batch_size,
                                              shuffle=True)

    # # 展示数据集
    # images, label = next(iter(train_loader))
    # print(images.shape)
    # print(label)
    #
    # images_example = torchvision.utils.make_grid(images)
    # images_example = images_example.numpy().transpose(1, 2, 0)
    mean = [.5, .5, .5]
    std = [.5, .5, .5]
    # images_example = images_example * std + mean
    # plt.imshow(images_example)
    # plt.show()

    # 创建模型实例
    model = AutoEncoder()
    if opt.device:
        model.cuda()
    # print(model)
    # summary(model, (1, 28, 28))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(opt.epochs):
        running_loss = 0.0
        print("Epoch {}/{}".format(epoch + 1, opt.epochs))
        print("-" * 60)

        for data in tqdm(train_loader):
            x_train = data[0]

            noisy_x_trian = x_train + 0.5 * torch.randn(*x_train.shape)
            noisy_x_trian = torch.clamp(noisy_x_trian, 0.0, 1.0)

            if opt.device:
                x_train, noisy_x_trian = Variable(x_train.cuda()), Variable(noisy_x_trian.cuda())
            else:
                x_train, noisy_x_trian = Variable(x_train), Variable(noisy_x_trian)

            train_pred = model(noisy_x_trian)
            loss = loss_fn(train_pred, x_train)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.data.item()

        print("Loss is:{:.4f}".format(running_loss / len(dataset_train)))

    # 测试部分
    x_test, _ = next(iter(tqdm(test_loader)))

    img_origin = torchvision.utils.make_grid(x_test)
    img_origin = img_origin.numpy().transpose(1, 2, 0)
    std = [.5, .5, .5]
    mean = [.5, .5, .5]
    img_origin = img_origin * std + mean

    plt.figure()
    plt.title("Origin Image")
    plt.imshow(img_origin)
    plt.show()

    # 打码
    noisy_x_test = img_origin + .5 * np.random.randn(*img_origin.shape)
    noisy_x_test = np.clip(noisy_x_test, 0.0, 1.0)

    plt.figure()
    plt.title("Mosaic Image")
    plt.imshow(noisy_x_test)
    plt.show()

    img_test = x_test + .5 * torch.randn(*x_test.shape)
    img_test = torch.clamp(img_test, 0.0, 1.0)

    if opt.device:
        img_test = Variable(img_test.cuda())
    else:
        img_test = Variable(img_test)

    test_pred = model(img_test)

    img_test = test_pred.data.view(-1, 1, 28, 28)

    img_test = torchvision.utils.make_grid(img_test)
    img_test = img_test.cpu().numpy().transpose(1, 2, 0)
    img_test = img_test * std + mean
    img_test = np.clip(img_test, 0.0, 1.0)
    plt.figure()
    plt.title("Fixed Image")
    plt.imshow(img_test)
    plt.show()

    # # 测试部分
    # x_test, _ = next(iter(test_loader))
    #
    # img_1 = torchvision.utils.make_grid(x_test)
    # img_1 = img_1.numpy().transpose(1, 2, 0)
    #
    # img_1 = img_1 * std + mean
    #
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title("Origin Image")
    # plt.imshow(img_1)
    #
    # # 打码
    # noisy_x_test = img_1 + .5 * np.random.randn(*img_1.shape)
    # noisy_x_test = np.clip(noisy_x_test, 0.0, 1.0)
    #
    # plt.subplot(1, 3, 2)
    # plt.title("Mosaic Image")
    # plt.imshow(noisy_x_test)
    #
    # img_2 = x_test + .5 * torch.randn(*x_test.shape)
    # img_2 = torch.clamp(img_2, .0, 1.)
    #
    # if opt.device:
    #     img_2 = Variable(img_2.view(-1, 28*28).cuda())
    # else:
    #     img_2 = Variable(img_2.view(-1, 28*28))
    #
    # test_pred = model(img_2)
    #
    # img_test = test_pred.data.view(-1, 1, 28, 28)
    #
    # img_2 = torchvision.utils.make_grid(img_test)
    # img_2 = img_2.cpu().numpy().transpose(1, 2, 0)
    # img_2 = img_2 * std + mean
    # img_2 = np.clip(img_2, 0., 1.)
    # plt.subplot(1, 3, 3)
    # plt.title("Fixed Image")
    # plt.imshow(img_2)
    # plt.show()


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output


if __name__ == "__main__":
    main()
