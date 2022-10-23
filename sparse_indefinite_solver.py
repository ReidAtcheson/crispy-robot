import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import arnoldi
def remap_quad(a,b,xs,ws):
    xsab = 0.5*(1.0+xs)*b-0.5*(-1.0+xs)*a
    wsab = ws * (b-a)/2.0
    return xsab,wsab,

def multi_interval_quad(a,b,c,d,xs,ws):
    xsab,wsab=remap_quad(a,b,xs,ws)
    xscd,wscd=remap_quad(c,d,xs,ws)
    xsout=np.concatenate([xsab,xscd])
    wsout=np.concatenate([wsab,wscd])
    return xsout,wsout


seed=239872 
rng=np.random.default_rng(seed)

mx=128
my=128
m=mx*my
A=sp.diags([rng.uniform(-2,2,size=m), rng.uniform(-2,2,size=m), rng.uniform(1,2,size=m)],[-mx,-1,0],shape=(m,m))
A=0.5*(A+A.T)


neval=0
def evalA(x):
    global neval
    neval=neval+1
    return A@x

eigA,v=spla.eigsh(spla.LinearOperator((m,m),matvec=evalA),which="LA")
eigA,v=spla.eigsh(spla.LinearOperator((m,m),matvec=evalA),which="SA")

print(eigA)
print(neval)
