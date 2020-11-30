import torch
import torch.nn as nn

class ChatNet(nn.Module):
  def __init__(self,input_size,classes):
    super().__init__()
    self.l1=nn.Linear(input_size,128)
    self.l2=nn.Linear(128,64)
    self.l3=nn.Linear(64,32)
    self.l4=nn.Linear(32,classes)
    self.d0=nn.Dropout(p=0.1)
    self.relu=nn.ReLU()


  def forward(self,x):
    out=self.l1(x)
    out=self.relu(out)
    out=self.l2(out)
    out=self.relu(out)
    out=self.l3(out)
    out=self.relu(out)
    out=self.d0(out)
    out=self.l4(out)

    return out
