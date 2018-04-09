import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import Function
import torch.optim as optim
from binary_modules import BinarizeLinear

BATCH_SIZE = 100

train_loader = DataLoader(
                datasets.MNIST(root='./mnist_data',train=True,download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=BATCH_SIZE,shuffle=True)

test_loader = DataLoader(
                datasets.MNIST(root='./mnist_data',train=False,download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=BATCH_SIZE,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.drop=nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)

def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.float().topk(maxk,1)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

model = Net()
model.cuda()

criterion = nn.NLLLoss()
criterion.cuda()
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.cuda(),target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            p.data.copy_(p.data.clamp_(-1,1))
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data,target = data.cuda(),target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 41):
    train(epoch)
    test()