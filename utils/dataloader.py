from inversefed import consts
import torch
from torchvision import datasets, transforms 

class DataLoader: 
    def __init__(self, data, device): 
        self.data = data 
        self.device = device
        
    def get_mean_std(self): 
        if self.data == 'cifar10': 
            mean, std = consts.cifar10_mean, consts.cifar10_std 
        elif self.data ==  'cifar100': 
            mean, std = consts.cifar100_mean, consts.cifar100_std 
        elif self.data == 'mnist': 
            mean, std = consts.mnist_mean, consts.mnist_std 
        elif self.data == 'imagenet':
            mean, std = consts.imagenet_mean, consts.imagenet_std 
        else: 
            raise Exception("dataset not found")
        return mean, std

    def get_data_info(self):
        mean, std = self.get_mean_std()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

        dm = torch.as_tensor(mean)[:, None, None].to(self.device)
        ds = torch.as_tensor(std)[:, None, None].to(self.device)
        data_root = 'data/cifar_data'
#         data_root = '~/.torch'
        if self.data == 'cifar10': 
            dataset = datasets.CIFAR10(root=data_root, download=True, train=False, transform=transform)
        elif self.data ==  'cifar100': 
            dataset = datasets.CIFAR100(root=data_root, download=True, train=False, transform=transform)
        elif self.data == 'mnist': 
            dataset = datasets.MNIST(root=data_root, download=True, train=False, transform=transform)
        elif self.data == 'imagenet':
            dataset = datasets.ImageNet(root=data_root, download=True, train=False, transform=transform)
        else: 
            raise Exception("dataset not found, load your own datasets")

        data_shape = dataset[0][0].shape 
        classes = dataset.classes 

        return dataset, data_shape, classes, (dm, ds)
    

