import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bisect


import arnoldi


seed=239872
rng=np.random.default_rng(seed)




m=20
A=rng.uniform(-1,1,size=(m,m))

for k in range(1,m+1):
    print(k)
    #k=9
    v=rng.uniform(-1,1,size=m)
    #v=v/np.linalg.norm(v)
    #alpha=0.2
    #A=A.T @ A+alpha*np.eye(m)
    A=0.5*(A+A.T)

    V,H=arnoldi.arnoldi(lambda x : A@x,v,k)

    y=rng.uniform(-1,1,size=k)


    V,H=arnoldi.arnoldi(lambda x : A@x,v,k)


    nsamples=100
    eigA=la.eigvalsh(A)
    zs=np.linspace(min(eigA),max(eigA),nsamples)
    x,p=arnoldi.gmres_step(lambda x : A@x,v,zs,k)
    _,p0=arnoldi.gmres_step(lambda x : A@x,v,np.array([0.0]),k)
    print(f"Pn(0.0) = {p0}")




    plt.plot(zs,p)
    plt.scatter(eigA,np.zeros(len(eigA)))

    ks=str(k).zfill(5)
    plt.savefig(f"plots/polynomial{ks}.svg")
    plt.close()
