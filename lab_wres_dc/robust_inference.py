import torch
import torch.nn.functional as F
from torch.autograd import Variable

def robust_inference(model, loader, args, target_model=False, note='None'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            try:
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
            except:
                test_loss += F.cross_entropy(output[-1], target, reduction='sum').item()
                pred = output[-1].data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(loader.dataset)
    if target_model:
        sr = 100. * float(correct) / len(loader.dataset)
        psr = sr
    else:
        sr = 100-100. * float(correct) / len(loader.dataset)
        psr = 100-sr
    print('<< {} >> Average loss: {:.4f}, Predict Success Rate: {}/{} ({:.2f}%)'.format(
         note, test_loss, correct, len(loader.dataset), psr))