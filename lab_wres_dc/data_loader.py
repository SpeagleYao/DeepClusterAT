from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def clean_loader_cifar(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    clean_loader= DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return clean_loader

def adv_loader_data(args, adv_samples, targets):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    adv_loader = DataLoader(TensorDataset(adv_samples, targets), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return  adv_loader











