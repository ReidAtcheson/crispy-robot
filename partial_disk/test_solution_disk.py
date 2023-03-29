import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bisect
import pickle
import tqdm


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

    return np.array(xs),np.array(ys)


def halfcircle(m):
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
        if np.hypot(x,y)>mineig and x>0:
            xs.append(x)
            ys.append(y)
    return xs,ys







mx=513
my=513
assert(mx%2==1)
assert(my%2==1)
xs,ys=np.meshgrid(np.linspace(-1,1,mx),np.linspace(-1,1,my))
xs=xs.flatten()
ys=ys.flatten()
zs=xs+1j*ys
nthetas=8

def A(u):
    return zs*u


v=zs.copy()

#order=64
for order in tqdm.tqdm(range(2,32)):
    V,H=arnoldi.arnoldi(A,v,order)
    Vh=V[:,:order]


    thetas=np.linspace(-np.pi,np.pi,nthetas)
    exact=np.zeros_like(zs)
    atzero=np.zeros_like(zs)
    onlyonce=False
    zid=0
    for i,(x,y) in enumerate(zip(xs,ys)):
        theta=np.arctan2(y,x)
        if np.hypot(x,y)<1e-6:
            assert(onlyonce==False)
            atzero[i]=1.0
            zid=i
            onlyonce=True
        if np.hypot(x,y)>mineig and theta>=thetas[4] and theta<=thetas[5]:
            exact[i]=1
    assert(onlyonce)


    soln=Vh@(Vh.conj().T@exact)
    plt.imshow(np.abs(1.0-soln).reshape((mx,my)),extent=[-1,1,-1,1],vmin=0.0,vmax=1.0)
    plt.title(f"order = {order}, P(0) = {np.abs(1.0-soln[zid])}")
    plt.colorbar()
    #Draw a unit circle on the above plot
    circle=plt.Circle((0,0),1.0,color='r',fill=False)
    plt.gca().add_artist(circle)
    plt.savefig(f"disk_{str(order).zfill(3)}.png")
    plt.close()

    plt.imshow(np.abs(1.0-soln).reshape((mx,my)),extent=[-1,1,-1,1],vmin=1.0,vmax=2.0)
    plt.title(f"order = {order}, P(0) = {np.abs(1.0-soln[zid])}")
    plt.colorbar()
    #Draw a unit circle on the above plot
    circle=plt.Circle((0,0),1.0,color='r',fill=False)
    plt.gca().add_artist(circle)
    plt.savefig(f"g1_disk_{str(order).zfill(3)}.png")
    plt.close()

    plt.imshow(np.log(np.abs(H)+1e-16))
    plt.colorbar()
    plt.savefig(f"H_{str(order).zfill(3)}.png")
    plt.close()







