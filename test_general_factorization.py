import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bisect
from scipy.special import roots_legendre


import arnoldi


seed=239872
rng=np.random.default_rng(seed)




m=20
k=10
A=rng.uniform(-1,1,size=(m,m))
v=rng.uniform(-1,1,size=m)
def dot(x,y):
    if len(x.shape)==2 and len(y.shape)==2 and ex.shape[1]==y.shape[1]:
        return np.dot(x,y)
    else:
        return x.T @ y



V,H=arnoldi.arnoldi_general(lambda x : A@x,v,k,dot)




order=20
xs,ws=roots_legendre(order)
def legdot(x,y):
    if len(x.shape)==2 and len(y.shape)==2 and ex.shape[1]==y.shape[1]:
        return np.dot(ws*x,y)
    else:
        return (ws*x.T) @ y


def timesx(y):
    return (xs*y.T).T

v=np.ones(len(ws))
#V,H=arnoldi.arnoldi_general(lambda x : xs*x,v,k,legdot)
V,H=arnoldi.arnoldi_general(timesx,v,k,legdot)


plt.plot(xs,V[:,0])
plt.plot(xs,V[:,1])
plt.plot(xs,V[:,2])
plt.plot(xs,V[:,3])
plt.savefig("test.svg")
plt.close()

print(np.linalg.norm(timesx(V[:,0:k])-V@H))

plt.imshow(H)
plt.savefig("H.png")


