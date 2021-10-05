import numpy as np
a=np.array([[1,2,3,1],[2,3,4,2]])
c=np.argsort(a)
mask=[1,1,0,1]
mm=np.random.choice(10,6)
b=np.argmax(a,1)
regularization_strengths = [(1+0.1*i)*1e4 for i in range(-3,4)]+[(5+0.1*i)*1e4 for i in range(-3,4)]
t={}
c=[[1,2,3],[5,6,7]]

print('a%i'% 7)