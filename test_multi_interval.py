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

order=50
for k in range(1,30):
    xs,ws=roots_legendre(order)
    print(f"processing {k}")


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





    a=-4.0
    b=-0.5
    c=0.5
    d=4.0
    xs,ws=multi_interval_quad(a,b,c,d,xs,ws)

    #print(sum(xs*xs*ws) - (4.0**3/3 - 1**3/3))





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
    x,y = arnoldi.gmres_general_step(timesx,v,k,legdot)


    beta=sum(ws)
    zs=np.linspace(min([a,b,c,d]),max([a,b,c,d]),300)
    P=arnoldi.arnoldi_general_basis(H,v,zs,beta)

    r=1.0 - zs*(P[:,0:k]@y)

    plt.plot(zs,r)
    plt.scatter([a,b,c,d],np.zeros(4))
    kstr=str(k).zfill(3)
    plt.savefig(f"res{kstr}.svg")
    plt.close()



#plt.plot(zs,P[:,0])
#plt.plot(zs,P[:,1])
#plt.plot(zs,P[:,2])
#plt.plot(zs,P[:,3])
#plt.savefig("test.svg")
#plt.close()
#
#print(np.linalg.norm(timesx(V[:,0:k])-V@H))
#
#plt.imshow(H)
#plt.savefig("H.png")


