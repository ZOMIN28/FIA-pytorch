import torch
import torch.nn.functional as F
import torch.utils.model_zoo
import torch.autograd as autograd



def vgg16(**kwargs):
    """
    VGGFace model.
    """
    model = Vgg16(**kwargs)
    return model


class Vgg16(torch.nn.Module):
    def __init__(self, classes=2622):
        """VGGFace model.

        Face recognition network.  It takes as input a Bx3x224x224
        batch of face images and gives as output a BxC score vector
        (C is the number of identities).
        Input images need to be scaled in the 0-1 range and then 
        normalized with respect to the mean RGB used during training.

        Args:
            classes (int): number of identities recognized by the
            network

        """
        super().__init__()
        self.conv1 = _ConvBlock(3, 64, 64)
        self.conv2 = _ConvBlock(64, 128, 128)
        self.conv3 = _ConvBlock(128, 256, 256, 256)
        self.conv4 = _ConvBlock(256, 512, 512, 512)
        self.conv5 = _ConvBlock(512, 512, 512, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def conv3_features(self,x):
        y = x
        y.retain_grad()
        z = self.conv1(y)
        z = self.conv2(z)
        z = self.conv3(z)
        return z
    
    def conv5_features(self,x):
        y = x
        #y.retain_grad()
        z = self.conv1(y)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.conv5(z)
        return z

    # 求解特征输出到conv3的梯度
    def features_grad(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.conv3(x)
        y.retain_grad()
        z = self.conv4(y)
        z = self.conv5(z)
        #z.backward(torch.ones_like(z))
        #y_grad = autograd.grad(z.sum(), y, retain_graph=True)[0]
        return z,y


class _ConvBlock(torch.nn.Module):
    """A Convolutional block."""

    def __init__(self, *units):
        """Create a block with len(units) - 1 convolutions.

        convolution number i transforms the number of channels from 
        units[i - 1] to units[i] channels.

        """
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_, out, 3, 1, 1)
            for in_, out in zip(units[:-1], units[1:])
        ])
        
    def forward(self, x):
        # Each convolution is followed by a ReLU, then the block is
        # concluded by a max pooling.
        for c in self.convs:
            x = F.relu(c(x))
        return F.max_pool2d(x, 2, 2, 0, ceil_mode=True)