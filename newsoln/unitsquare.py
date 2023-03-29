import arnoldi
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm


mx=129
my=128
order=128
for order in tqdm.tqdm(range(8,256)):
    xs,ys=np.meshgrid(np.linspace(-1,1,mx),np.linspace(-1,1,my))
    shape=xs.shape
    xs=xs.flatten()
    ys=ys.flatten()
    zs=xs+1j*ys


    def A(u):
        return zs*u


    v=np.ones_like(zs)
    exact=np.zeros_like(zs)
    exact[xs>0]=1.0
    exact[xs<=0]=0.0

    V,H=arnoldi.arnoldi(A,v,order)
    Vh=V[:,0:order]

    #print(Vh.shape)
    #print(np.linalg.norm(Vh.conj().T @ Vh - np.eye(Vh.shape[1])))



    plt.imshow(np.abs(1.0 - Vh @(Vh.conj().T @ exact)).reshape(shape),extent=[-1,1,-1,1],vmin=0.0,vmax=1.0)
    plt.title(f"Potential residual polynomial order {order}")
    plt.colorbar()
    plt.savefig(f'out_{str(order).zfill(3)}.png')
    plt.close()
