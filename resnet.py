import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, obs_channels, state_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(obs_channels, state_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(state_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(state_channels, state_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(state_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or obs_channels != state_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(obs_channels, state_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(state_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, obs_channels,gridH,gridW,actionspace,state_channels,state_action_channels):
        super(ResNet18, self).__init__()
        self.obs_channels = obs_channels
        self.rep_conv = nn.Conv2d(obs_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pred_conv = nn.Conv2d(state_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dyn_conv = nn.Conv2d(state_action_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock, 256, 2, stride=1),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            )
        self.hc = nn.Linear(256, state_channels)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * gridH * gridW, 128),
            nn.ReLU(),
            nn.Linear(128, actionspace)         # Policy 
            )
        self.fc2 = nn.Sequential(
            nn.Linear(256 * gridH * gridW, 128),
            nn.ReLU(),
            nn.Linear(128, 1)         # Value 
            )
        self.gc1 = nn.Linear(256, state_channels)
        self.gc2 = nn.Sequential(
            nn.Flatten(), #Output
            nn.Linear(256*gridH*gridW,128),
            nn.ReLU(),
            nn.Linear(128,1) #reward
            )      
        
    def _make_layer(self, block, state_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.obs_channels, state_channels, stride))
            self.obs_channels = state_channels
        return nn.Sequential(*layers)

    def h(self, x):
        #x = observation
        #out = state
        out = self.rep_conv(x)
        out = self.resnet(out)
        return self.hc(out)
    
    def f(self,x):
        #x = state
        #out = policy,value
        out = self.pred_conv(x)
        out = self.resnet(out)
        out = nn.Flatten()(out)
        return self.fc1(out),self.fc2(out)
    
    def g(self,x):
        #x = state || action
        #return nextstate,reward
        out = self.dyn_conv(x)
        out = self.resnet(out)
        return self.gc1(out),self.gc2(out)
    
#concat or add channels for history obs?
#create different output layers for f(state)->(policy,value),g(action,state)->(state,reward),h(obs)->(state)
# change fc layer after avgpool ofc and avg pooling or maxpool? and what size? read paper..
#so input plane for h is history(8)*channel of each obs(5)+action_space((2*dots*(dots-1))*2+1)
#that will give me a state of 256 hidden channels for each i think with 3x3 kernels using 16 residual blocks
#g will recieve (state)256 +(concat with) actions(-) as input 
#that will give me 256(state),1(reward)
#f will recieve (state)256 as input
#that will give me action_space(4*dots*(dots-1)+1) , value(1)