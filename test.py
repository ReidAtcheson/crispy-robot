import numpy as np
import scipy.linalg as la
from sympy import symbols


z=symbols("z")
p=np.asarray([z for _ in range(0,6)])
p[0]=1.0
p[1]=p[0]*z
p[2]=p[1]*z

print(p)
