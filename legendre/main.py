import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


mx=128
my=128
m=mx*my
xs,ys=np.meshgrid(np.linspace(-1,1,mx),np.linspace(-1,1,my))
xs=xs.flatten()
ys=ys.flatten()
zs=np.array([x+1j*y if np.hypot(x,y)<=1 else 0.0 for x,y in zip(xs,ys)])

order=16
P=np.zeros((mx*my,order+1),dtype=np.complex128)

P[:,0]=1.0
P[:,1]=zs
for i in range(1,order):
    P[:,i+1]=2*zs*P[:,i] - P[:,i-1]


plt.imshow(np.abs(P[:,-1]).reshape((mx,my)),extent=[-1,1,-1,1])
plt.colorbar()
plt.savefig("chebyshev.png")
