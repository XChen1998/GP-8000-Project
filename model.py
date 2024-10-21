import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # Output channels expansion factor

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both conv1 and downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample  # For adjusting dimensions when needed
        self.stride = stride

    def forward(self, x):
        identity = x  # Store the input tensor

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # Apply ReLU

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out += identity  # Add residual connection
        out = self.relu(out)  # Apply ReLU

        return out

class Bottleneck(nn.Module):
    expansion = 4  # Output channels expansion factor

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)

        # 3x3 conv
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)

        # 1x1 conv to restore dimensions
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample  # For adjusting dimensions when needed
        self.stride = stride

    def forward(self, x):
        identity = x  # Store the input tensor

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # Apply ReLU

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)  # Apply ReLU

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out += identity  # Add residual connection
        out = self.relu(out)  # Apply ReLU

        return out

def get_resnet_config(depth):
    if depth == 18:
        return BasicBlock, [2, 2, 2, 2]
    elif depth == 34:
        return BasicBlock, [3, 4, 6, 3]
    elif depth == 50:
        return Bottleneck, [3, 4, 6, 3]
    elif depth == 101:
        return Bottleneck, [3, 4, 23, 3]
    elif depth == 152:
        return Bottleneck, [3, 8, 36, 3]
    else:
        raise ValueError("Invalid depth. Only ResNet-18, 34, 50, 101, and 152 are supported.")

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        block, layers = get_resnet_config(depth)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # Number of input channels to the first layer
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # Each element in the list indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element list, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # First convolutional layer and max pooling
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Initialize batchnorm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Create one layer (multiple blocks)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            # Need downsampling to match dimensions
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Implement forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x