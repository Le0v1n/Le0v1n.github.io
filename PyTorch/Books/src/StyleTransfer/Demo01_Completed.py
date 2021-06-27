import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ]
)

def loading(path = None):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img

content_img = loading("images/2.jpg")
content_img = Variable(content_img).cuda()  # torch.Size([1, 3, 224, 224])

style_img = loading("images/5.jpg")
style_img = Variable(style_img).cuda()  # torch.Size([1, 3, 224, 224])

class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        # target是通过卷积获取到的输入图像中的内容
        # weight是我们设置的一个权重, 用来控制内容和风格对最终合成图像的影响程度
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight  # .detach()是对提取到的内容进行梯度锁定 -> 不计算梯度
        # 指定损失函数类型
        self.loss_fn = torch.nn.MSELoss()

    # 前向传播 -> 用于计算输入图像和内容图像中间的损失值
    def forward(self, input):
        # input 代表输入图像
        self.loss = self.loss_fn(input * self.weight, self.target)
        return input

    # 后向传播 -> 根据计算得到的损失值进行后向传播, 并返回损失值
    def backward(self):
        self.loss.backward(retain_graph=True)
        """
        retrain_graph = True
                一般情况下, 每次backward()时, 默认会把这个计算图释放掉, 所以一般来说, 每次迭代只需要一次forward()和一次
            backward(), 即forward()和backward()是成对出现的. 一般一次backward()也是够用的, 但是不排除由于自定义loss等造成的
            复杂性, 需要一次forward(), 多个不同的loss的backward()来累积同一个网络的grad, 以此来实现参数的更新.
                于是若在当前backward()之后, 不执行forward()而是执行另一个backward(), 需要在当前的backward()时使用
            retrain_graph = True指定保留计算图.
        """
        return self.loss


class Gram_matrix(torch.nn.Module):
    """
        我们通过卷积神经网络提取了风格图片的风格, 这些风格其实是由数字组成的, 数字的大小代表了图片中风格的突出程度, 而Gram矩阵是
        矩阵的内积运算, 在运算过后输入到该矩阵的特征图中的大的数字会变得更大, 这就相当于图片的风格被放大了, 放大的风格再参与损失
        计算,便能够对最后的合成图片产生更大的影响.
    """
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())  # .t表示矩阵的转置
        return gram.div(a * b * c * d)


class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)  # in-place
        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

cnn = models.vgg16(pretrained=True).features
# print(cnn)
# print("-" * 100)

if torch.cuda.is_available():
    cnn = cnn.cuda()

model = copy.deepcopy(cnn)

# summary(cnn, (3, 224, 224))

content_losses = []
style_losses = []

content_weight = 1
style_weight = 1000

# 创建一个新的模型
new_model = torch.nn.Sequential()

gram = Gram_matrix()

if torch.cuda.is_available():
    new_model = new_model.cuda()
    gram = gram.cuda()

content_layer = ["Conv_3"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
index = 1

for layer in list(model)[:8]:
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_" + str(index)
        new_model.add_module(name, layer)
        if name in content_layer: # content_layer = ["Conv_3"]
            target = new_model(content_img).clone()
            content_loss = Content_loss(content_weight, target)
            new_model.add_module("content_loss_" + str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer: # style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss_" + str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = "ReLU_" + str(index)
        new_model.add_module(name, layer)
        index += 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = "MaxPool_" + str(index)
        new_model.add_module(name, layer)

print(new_model)

input_img = content_img.clone()
print(input_img.shape)
parameter = torch.nn.Parameter(input_img.data)  # 将一个不可训练的tensor转换成可训练的类型parameter
print(parameter.shape)
optimizer = torch.optim.LBFGS([parameter])

epoch_n = 800
epoch = 0
time_open = time.time()

while epoch <= epoch_n:

    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()
        global epoch
        epoch += 1
        if epoch % 50 == 0:
            print("Epoch:{} Style Loss:{:4f} Content Loss:{:4f}".format(epoch, style_score.item(), content_score.item()))
        return style_score + content_score

    optimizer.step(closure)

time_end = time.time()
print("程序运行时间为: {}s".format(int(time_end - time_open)))

plt.figure("Res")
plt.subplot(1, 3, 1)
plt.title("Content Image")
plt.imshow(content_img.data.cpu().squeeze(0).numpy().transpose([1, 2, 0]))

plt.subplot(1, 3, 2)
plt.title("Style Image")
plt.imshow(style_img.data.cpu().squeeze(0).numpy().transpose([1, 2, 0]))

plt.subplot(1, 3, 3)
plt.title("Style transfer Image")
plt.imshow(parameter.data.cpu().squeeze(0).numpy().transpose([1, 2, 0]))
plt.show()
