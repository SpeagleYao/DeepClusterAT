from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from loss import *
import sys
sys.path.append('..')
from models import *
from util import AverageMeter, Logger, UnifLabelSampler
from utils import Logger
from random import shuffle
from tqdm import tqdm
import time
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 model Train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=499, metavar='N',
                help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='log_ft_s')
parser.add_argument('--nmb_cluster', '--k', type=int, default=100,
                    help='number of cluster for k-means (default: 100)')
args = parser.parse_args()

def attack(model, img, label, eps=0.031, iters=10, step=0.007):
    adv = img.detach()
    adv.requires_grad = True

    for j in range(iters):
        out_adv = model(adv.clone())
        loss = F.cross_entropy(out_adv, label)
        loss.backward()

        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    return adv.detach()

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

args.cuda = not args.no_cuda and torch.cuda.is_available()

log_filename = 'res18n_ft_adv_k{}.txt'.format(args.nmb_cluster)
sys.stdout = Logger(os.path.join(args.save_dir, log_filename))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    print("No cuda participate.")

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = ResNet18()
fd = int(model.top_layer.weight.size()[1]) # [10, 512]
model.top_layer = None
model.features = nn.DataParallel(model.features)
model.cuda()
if args.cuda:
    model.cuda()

model_checkpoint = '../cp_dc_pt/res18_pt_adv_k'+str(args.nmb_cluster)+'.pth'
model.load_state_dict(torch.load(model_checkpoint))

model.top_layer = nn.Linear(fd, 10)
model.top_layer.weight.data.normal_(0, 0.01)
model.top_layer.bias.data.zero_()
model.top_layer.cuda()

optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
criterion = LabelSmoothing(0.1)


def run_dc(epoch):

    loss = train(train_loader, model, criterion, optimizer, epoch)
    print(f'Train Epoch: {epoch}:\tLoss: {loss}')


def train(loader, model, crit, opt, epoch):

    losses = AverageMeter()
    model.train()
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        model.eval()
        adv_sample = attack(model, data, target)
        data = torch.cat((data, adv_sample), 0)
        target = torch.cat((target, target), 0)
        rg = list(range(len(data)))
        shuffle(rg)
        data = data[rg]
        target = target[rg]

        model.train()
        opt.zero_grad()
        optimizer_tl.zero_grad()

        output = model(data)
        loss = crit(output, target)

        # record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        loss.backward()
        opt.step()
        optimizer_tl.step()
        
    filename = '../cp_dc_ft/res18_ft_adv_k{}.pth'.format(args.nmb_cluster)
    torch.save(model.state_dict(), filename)

    return losses.avg



def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))


def compute_features(dataloader, model, N):
    
    model.eval()

    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch_size: (i + 1) * args.batch_size] = aux
            else:
                # special treatment for final batch
                features[i * args.batch_size:] = aux

    return features


for epoch in range(1, args.epochs + 1):
    if epoch in [15, 20]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print("Learning rate reduced.")
    run_dc(epoch)
    inference()