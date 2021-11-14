import numpy as np
# a=np.random.choice(range(4),p=[0.05,0.25,0.6,0.1])
xx=np.array([1,2,3])
def f(x,a):
    return np.exp(a*x)/np.sum(np.exp(a*x))
print(f(xx,5),f(xx,1),f(xx,0.1))
>>[4.50940412e-05 6.69254912e-03 9.93262357e-01]
>>[0.09003057 0.24472847 0.66524096]
>>[0.30060961 0.33222499 0.3671654 ]
