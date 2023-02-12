import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

# # Original CAM code
# class CAM(Net):

#     def __init__(self):
#         super(CAM, self).__init__()

#     def forward(self, x):

#         x = self.stage1(x)

#         x = self.stage2(x)

#         x = self.stage3(x)

#         x = self.stage4(x)

#         x = F.conv2d(x, self.classifier.weight)
#         x = F.relu(x)

#         x = x[0] + x[1].flip(-1)

#         return x

    
class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, clf=True):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        if clf:
            x = torchutils.gap2d(x, keepdims=True)
            x = self.classifier(x)
            x = x.view(-1, 20)
        else:
            x = F.conv2d(x, self.classifier.weight)
            x = F.relu(x)
            x = x[0] + x[1].flip(-1)
        return x
    
    
class CAM_with_hook(Net):

    def __init__(self):
        super(CAM_with_hook, self).__init__()
        
        # placeholder for the gradients
        self.gradients_layer4 = None
        self.gradients_layer3 = None
        self.gradients_layer2 = None
        self.gradients_layer1 = None
    
    # hook for the gradients of the activations
    def activations_hook_layer4(self, grad):
        self.gradients_layer4 = grad
        
     # hook for the gradients of the activations
    def activations_hook_layer3(self, grad):
        self.gradients_layer3 = grad
    
     # hook for the gradients of the activations
    def activations_hook_layer2(self, grad):
        self.gradients_layer2 = grad
        
     # hook for the gradients of the activations
    def activations_hook_layer1(self, grad):
        self.gradients_layer1 = grad
        
        
    def forward(self, x, clf=True):
        x = self.stage1(x)
        # register the hook
        h = x.register_hook(self.activations_hook_layer1)

        x = self.stage2(x)
        # register the hook
        h = x.register_hook(self.activations_hook_layer2)

        x = self.stage3(x)
        # register the hook
        h = x.register_hook(self.activations_hook_layer3)

        x = self.stage4(x)
        # register the hook
        h = x.register_hook(self.activations_hook_layer4)

        if clf:
            x = torchutils.gap2d(x, keepdims=True)
            x = self.classifier(x)
            x = x.view(-1, 20)
        else:
            # B = x.shape[0]//2
            x = F.conv2d(x, self.classifier.weight)
            x = F.relu(x)
            x = x[0] + x[1].flip(-1)
            # x = x[:B] + x[B:].flip(-1)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self, layer='layer4'):
        if layer=='layer1':
            return self.gradients_layer1
        elif layer=='layer2':
            return self.gradients_layer2
        elif layer=='layer3':
            return self.gradients_layer3
        else:
            return self.gradients_layer4
    
    # method for the activation exctraction
    def get_activations(self, x, layer='layer4'):
        x = self.stage1(x)
        if layer == 'layer1':
            return x
        x = self.stage2(x)
        if layer == 'layer2':
            return x
        x = self.stage3(x)
        if layer == 'layer3':
            return x
        x = self.stage4(x)
        return x