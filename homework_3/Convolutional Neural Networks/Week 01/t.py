import numpy as np
a=np.array([[1,2,3],[2,3,4]])
# a=np.pad(a,((1,1),(1,2)),constant_values=((0,0,0,0)))
print(np.max(a))