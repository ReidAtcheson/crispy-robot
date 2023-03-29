import numpy as np
import scipy.linalg as la
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


m=50
#Simple definite matrix first (all eigenvalues in (a,b))
a,b=-1.0,-0.5
c,d=0.5,4.0

D=rng.uniform(-1,1,size=(m,m))
Q,_=la.qr(D)
D=np.diag(list(np.linspace(a,b,m//2))+list(np.linspace(c,d,m//2)))
A=Q.T @ D @ Q



#Pick arbitrarily large order just so we don't lose any quadrature exactness
order=50
#Quadrature nodes,weights in [-1,1]
xs,ws=roots_legendre(order)

#Remap to a,b
xs,ws=multi_interval_quad(a,b,c,d,xs,ws)

#Desired order of iteration polynomial
k=20

#integral(x*y) over [a,b] via the above quadrature
def legdot(x,y):
    if len(x.shape)==2 and len(y.shape)==2 and ex.shape[1]==y.shape[1]:
        return np.dot(ws*x,y)
    else:
        return (ws*x.T) @ y

#Form the polynomial Pk of order `k` such that Pk(0)=1 and we minimize integral(Pk*Pk) over (a,b)
#This is represented by `y`
v=np.ones(len(ws))
beta=np.sqrt(legdot(v,v))
V,H=arnoldi.arnoldi_general(lambda y : (xs*y.T).T,v,k,legdot)
x,y = arnoldi.gmres_general_step(lambda y : (xs*y.T).T,v,k,legdot)

plt.imshow(H)
plt.colorbar()
plt.savefig("H.png")
plt.close()

zs=np.linspace(a,d,300)
P=arnoldi.arnoldi_general_basis(H,np.ones(len(zs)),zs,np.sqrt(np.sum(ws)))

plt.plot(xs,v-xs*(V[:,0:k]@y))
plt.savefig("polynomial0.svg")
plt.close()

plt.plot(zs,np.ones(len(zs))-zs*(P[:,0:k]@y))
plt.savefig("polynomial1.svg")
plt.close()

plt.plot(zs,(P[:,0:k]@y))
plt.savefig("polynomial2.svg")
plt.close()












#b=rng.uniform(-1,1,size=m)
b=np.array(np.linspace(a,b,m))
#Now do a few steps of polynomial iteration
x=np.zeros(m)
maxit=100
mres=[]
mres.append(np.linalg.norm(b))
for it in range(0,maxit):
    x,r=arnoldi.polynomial_step(lambda u : A@u,b,x,H,y,beta)
    #print(f"{np.linalg.norm(r)}")
    mres.append(np.linalg.norm(r))
    if mres[-1]<1e-14:
        break






res=[]
def callback(xk):
    global res
    #print(f"GMRES residual: {np.linalg.norm(b-A@xk)}")
    res.append(np.linalg.norm(b-A@xk))



spla.gmres(A,b,restart=k,callback=callback,callback_type="x",maxiter=maxit,tol=1e-14)
print(f"POLY({k}): {mres[-1]}")
print(f"GMRES({k}): {res[-1]}")
print(f"norm(POLY({k})): {np.sqrt(legdot(v-xs*(V[:,0:k]@y),v-xs*(V[:,0:k]@y)))}")




plt.semilogy(res)
plt.semilogy(mres)
plt.xlabel("Restarts")
plt.ylabel("Residual")
plt.legend([f"GMRES({k})",f"POLY({k})"])
plt.savefig("gmres.svg")

