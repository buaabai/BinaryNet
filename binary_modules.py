import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

def Binarize(tensor):
    return tensor.sign()

class BinarizeLinear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(BinarizeLinear,self).__init__(*args,**kwargs)
    def forward(self,input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        self.weight.data = Binarize(self.weight.data)
        out = nn.functional.linear(input,self.weight)
        if not self.bias is None:
            out += self.bias.view(1,-1).expand_as(out)
        return out

            
