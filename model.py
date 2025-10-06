import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3) # Conv 1 layer
        self.bn1 = nn.BatchNorm2d(16) # BatchNorm layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Conv 2 layer
        self.bn2 = nn.BatchNorm2d(32) # BatchNorm layer
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)  # Conv 3 layer
        self.bn3 = nn.BatchNorm2d(48) # BatchNorm layer
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1 ) # Conv 4 layer
        self.bn4 = nn.BatchNorm2d(64) # BatchNorm layer
        self.conv5 = nn.Conv2d(64,80, kernel_size=3, padding=1)  # Conv 5 layer
        self.relu = nn.ReLU()# ReLU layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Avgpool layer
        self.fc = nn.Linear(80, 10) # Linear layer
    
    def forward(self, x, intermediate_outputs=False):
        # TODO: 按照题目描述中的示意图计算前向传播的输出，如果 intermediate_outputs 为 True，也返回各个卷积层的输出。
        conv1_out=self.conv1(x)
        bn1=self.bn1(conv1_out)
        relu1=self.relu(bn1)
       

        conv2_out=self.conv2(relu1)
        bn2=self.bn2(conv2_out)
        relu2=self.relu(bn2)
        pool2=self.maxpool(relu2)

        conv3_out=self.conv3(pool2)
        bn3=self.bn3(conv3_out)
        relu3=self.relu(bn3)
        pool3=self.maxpool(relu3)

        conv4_out=self.conv4(pool3)
        bn4=self.bn4(conv4_out)
        relu4=self.relu(bn4)
        pool4=self.maxpool(relu4)

        conv5_out=self.conv5(pool4)
        relu5=self.relu(conv5_out)
        pool_average=self.avgpool(relu5)
        final_out=self.fc(pool_average.view(pool_average.size(0), -1))
        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out,conv3_out, conv4_out, conv5_out]
        else:
            return final_out
