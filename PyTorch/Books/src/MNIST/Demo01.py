import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# transform=transforms.ToTensor():将图像转化为Tensor，在加载数据的时候，就可以对图像做预处理
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
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

images, labels = next(iter(data_loader_train))  # iter()对()内的对象进行迭代, next(iter对象) 返回迭代器(iter对象)的下一个项目
img = torchvision.utils.make_grid(images)
"""
make_grid的作用是将若干幅图像拼成一幅图像。
其中padding的作用就是子图像与子图像之间的pad有多宽。nrow默认一行放入8张图片
def make_grid(tensor: Union[Tensor, list[Tensor]],
              nrow: int = 8,
              padding: int = 2,
              normalize: bool = False,
              value_range: Optional[tuple[int, int]] = None,
              scale_each: bool = False,
              pad_value: int = 0,
              **kwargs: Any) -> Tensor
"""
print(img.shape)  # torch.Size([3, 242, 242])
# 因为img原本的shape是[3, 242, 242], 这样的图片我们是不能直接imshow的,
# 因为不符合图像的排列规则, 所以我们要转换为[242, 242, 3]这样的形状
img = img.numpy().transpose(1, 2, 0)

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]

img = img * std + mean

# print([(labels[i]).data for i in range(64)])  # 如果我们不是用.item而是用.data或者不写, 那么打印出来的数据为: [tensor(3), tensor(8),..., tensor(5), tensor(2)]
print([labels[i].item() for i in range(64)])

plt.imshow(img)
plt.show()  # 切记不要忘了!
