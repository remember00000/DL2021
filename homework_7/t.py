import numpy as np
import torch

# a=np.random.randint(0,11,10)
# aa=torch.LongTensor(a)
# m = torch.nn.Sigmoid()
# input = torch.randn(2)
# output = m(input)
# aa=torch.randn(10,5)
# print('111')
a=torch.Tensor([[1,2,3],[4,5,6]])
# b=torch.cuda.FloatTensor(np.randn(2))
print(a)
print(a.squeeze())