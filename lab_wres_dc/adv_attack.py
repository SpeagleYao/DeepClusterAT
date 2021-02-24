import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def fgsm(img, label, model, criterion=F.cross_entropy, eps=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    out = model(adv)
    try:
        loss = criterion(out, label)
    except:
        loss = criterion(out[-1], label)
    loss.backward()

    noise = adv.grad

    if target_setting:
        adv.data = adv.data - eps * noise.sign()
    else:
        adv.data = adv.data + eps * noise.sign()
    adv.data.clamp_(0.0, 1.0)

    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()
    # print(f'fgsm:{l2}')
    return adv.detach(), l2


def pgd(img, label, model, criterion=F.cross_entropy, eps=0.03, iters=10,step=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)
        try:
            loss = criterion(out_adv, label)
        except:
            loss = criterion(out_adv[-1], label)
        loss.backward()

        noise = adv.grad
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()
    # print(f'pgd10:{l2}')
    return adv.detach(),l2


def mim(img, label, model, criterion=F.cross_entropy, eps=0.03, iters=10, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    iterations = iters
    step = eps / iterations
    noise = 0

    for j in range(iterations):
        out_adv = model(adv)
        try:
            loss = criterion(out_adv, label)
        except:
            loss = criterion(out_adv[-1], label)
        loss.backward()

        adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
        adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
        adv.grad = adv.grad / adv_mean
        noise = noise + adv.grad

        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2



def cw(inputs, targets, model, iters=1000, kappa=0, c=5, lr=1e-2, target_setting=False):

    def atanh(x):
        return 0.5*torch.log((1+x)/(1-x))

    def f(x):
        outputs = model(x)[-1]
        outputs = torch.tensor(outputs).cuda()
        one_hot_labels = torch.eye(len(outputs[0]))[targets].cuda()
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        if target_setting:
            return torch.clamp(i - j, min=-kappa)
        else:
            return torch.clamp(j - i, min=-kappa)

    # w = torch.zeros_like(inputs, requires_grad=True).cuda()
    w = (atanh(2*inputs-1) + 0.2*torch.randn_like(inputs)).cuda()
    w.requires_grad = True
    optimizer = optim.Adam([w], lr=lr)
    prev = 1e10
    for step in range(iters):
        a = 1 / 2 * (nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, inputs)
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
       # Early Stop when loss does not converge.
        if step % (iters // 50) == 0:
            # print(f'step: {step}\tl2: {torch.norm((a - inputs).reshape(inputs.shape[0], -1), dim=1).mean()}')
            if cost > prev:
                # print('Attack Stopped due to CONVERGENCE....')
                l2 = torch.norm((a - inputs).reshape(inputs.shape[0], -1), dim=1).mean()
                # print(f'cw:{l2}')
                return a, l2
            prev = cost

    adv = 1 / 2 * (nn.Tanh()(w) + 1)
    l2 = torch.norm((adv- inputs).reshape(inputs.shape[0], -1), dim=1).mean()
    print(f'cw:{l2}')
    return adv, l2

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def cw2(img, label, model, criterion=CWLoss(num_classes=10), eps=0.03, iters=10,step=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)[-1]
        loss = criterion(out_adv, label)

        loss.backward()

        noise = adv.grad
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2

class OursLoss(nn.Module):
    def __init__(self):
        super(OursLoss, self).__init__()
        self.alpha = 0.5
        self.delta = 0.1
        return

    def forward(self, w, output, target):
        
        criterion_weight = WeightLoss(args.delta)
        criterion_l2 = L2Loss(args.delta)

        loss_l2 = criterion_l2(output, target) #intra
        loss_w = criterion_weight(w) #inter

        loss = loss_l2 + args.alpha * loss_w

        return loss

def ours_adv(img, label, model, criterion=OursLoss(), eps=0.03, iters=10,step=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        w, out_adv = model(adv)
        loss = criterion(w, out_adv, label)

        loss.backward()

        noise = adv.grad
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2