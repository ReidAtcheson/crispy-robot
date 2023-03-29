import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bisect


import arnoldi


seed=239872
rng=np.random.default_rng(seed)




k=9
m=100
v=rng.uniform(-1,1,size=m)
v=v/np.linalg.norm(v)
A=rng.uniform(-1,1,size=(m,m))

alpha=0.2
A=A.T @ A+alpha*np.eye(m)
#A=0.5*(A+A.T)

#V,H=arnoldi.arnoldi(lambda x : A@x,v,k)
V,H=arnoldi.arnoldi(lambda x : A@x,v,k)


nsamples=100
eigA=la.eigvalsh(A)
zs=np.linspace(0.0,max(eigA),nsamples)


P=arnoldi.arnoldi_basis(H,v,zs)

x,p=arnoldi.gmres_step(lambda x : A@x,v,zs,k)
#_,p0=arnoldi.gmres_step(lambda x : A@x,v,np.array([0.0]),k)




plt.plot(zs,p)
plt.scatter(eigA,np.zeros(len(eigA)))
plt.savefig("solution.svg")
#for i in range(0,k+1):
#    plt.plot(zs,P[:,i])
#    plt.scatter(eigA,np.zeros(len(eigA)))
#
#    ks=str(i).zfill(5)
#    plt.savefig(f"plots_basis/polynomial{ks}.svg")

#    plt.close()
