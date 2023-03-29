import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bisect
import pickle


import arnoldi


seed=239872
rng=np.random.default_rng(seed)

mineig=1e-1

def fillcircle(m):
    n = m
    radius = np.sqrt(np.arange(n) / float(n)) 
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n) 
    points = np.zeros((n, 2))
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)
    points *= radius.reshape((n, 1))

    xs,ys=[],[]
    for x,y in zip(points[:,0],points[:,1]):
        if np.hypot(x,y)>mineig:
            xs.append(x)
            ys.append(y)
    return xs,ys






m=512
xs,ys=fillcircle(m)
xs=np.array(xs)
ys=np.array(ys)
zs=xs+1j*ys
m=len(zs)
D=np.diag(zs)
Q,_=la.qr(rng.standard_normal((m,m)))
A=Q @ D @ Q.T
v=Q@np.ones(m)
v=v/np.linalg.norm(v)



#V,H=arnoldi.arnoldi(lambda x : A@x,v,k)


k=500
#V,H=arnoldi.arnoldi(lambda x : A@x,v,k)


mx=512
my=512
nsamples=mx*my
X,Y=np.meshgrid(np.linspace(-1.1,1.1,mx),np.linspace(-1.1,1.1,my))
Z=X+1j*Y
Z=Z.flatten()

#x,p=arnoldi.gmres_step(lambda x : A@x,v,Z,k)
#_,p0=arnoldi.gmres_step(lambda x : A@x,v,np.array([0.0]),k)


eigA=la.eigvals(A)
plt.scatter(eigA.real,eigA.imag,s=1)

#Load p from file
with open("p.pickle","rb") as f:
    p=pickle.load(f)

plt.imshow(np.abs(p.reshape((mx,my))),extent=[-1.1,1.1,-1.1,1.1],vmin=0,vmax=1)
plt.colorbar()
plt.savefig("solution.svg")
