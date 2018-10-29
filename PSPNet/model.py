import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

pspnet_specs = {
    'n_classes': 3  # background, hand, object
}

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                            padding=padding, stride=stride, bias=bias, dilation=dilation)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]
        m=min(h,w)
        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(m/pool_size), int(m/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.final = nn.Sequential(
                        conv2DBatchNormRelu(4096, 512, 3, 1, 1, False),
                        conv2DBatchNormRelu(512, 128, 3, 1, 1, False),
                        conv2DBatchNormRelu(128, 64, 3, 1, 1, False)
                     )
    def forward(self, x):
        inp_shape = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)
        x = self.final(x)  
        
        return x,inp_shape

class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        self.n_classes = pspnet_specs['n_classes']                
        self.classification = nn.Conv2d(64, self.n_classes, 1, 1, 0)

    def forward(self, x, inp_shape):
        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')
        return x
