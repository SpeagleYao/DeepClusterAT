from models import *
from thop import profile

model = ResNet18()
fd = int(model.top_layer.weight.size()[1]) # [10, 512]
model.top_layer = None
# model.features = nn.DataParallel(model.features)
# model.cuda()
# if args.cuda:
    # model.cuda()

model.top_layer = nn.Linear(fd, 10)
model.top_layer.weight.data.normal_(0, 0.01)
model.top_layer.bias.data.zero_()
# model.top_layer.cuda()

input = torch.randn(1, 3, 32, 32)
flops, params = profile(model, inputs=(input, ))
print(flops, params)