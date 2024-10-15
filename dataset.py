from torchvision import datasets, transforms
from torch.utils.data import DataLoader


''' transform'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.RandomHorizontalFlip(p=0.2),
    # transforms.RandomGrayscale(p=0.2),
    # transforms.RandomAutocontrast(p=0.2),
    # transforms.RandomRotation(degrees=(0, 360))
])


''' dataset/dataloader '''

def load_cifar10(batch_size):
    # dataset
    d_tr = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
    d_ts = datasets.CIFAR10(root='./data', download=True, train=False, transform=transforms.ToTensor())

    # dataloader
    dl_tr = DataLoader(d_tr, batch_size=batch_size, shuffle=True, pin_memory=True) # pin_memory, pin_memory_device : gpu로드 속도를 높인다는데 어렵다.. 없는게 더 빠르다
    dl_ts = DataLoader(d_ts, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dl_tr, dl_ts