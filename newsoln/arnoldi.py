import numpy as np
import scipy.linalg as la
from sympy import symbols





#Factorizes A as
#A * V[:,0:k] = V * H
#Where V spans a Krylov space of A
#and V is orthogonal in the standard l2 inner product
def arnoldi(A,v,k):
    dtype=A(v).dtype
    norm=np.linalg.norm
    dot=np.vdot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    #Make V a matrix of size m x k+1 with same datatype as A
    V=np.zeros((m,k+1),dtype=dtype)
    H=np.zeros((k+1,k),dtype=dtype)
    #V[:,0]=v/norm(v)
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])

        h=V[:,0:j+1].conj().T @ w
        f=w-V[:,0:j+1] @ h

        s = V[:,0:j+1].conj().T @ f

        f = f - V[:,0:j+1] @ s

        h = h + s
        beta=norm(f)
        #H[j+1,j]=beta
        H[0:j+1,j]=h
        H[j+1,j]=beta
        #V[:,j+1]=f/beta
        V[:,j+1]=f.flatten()/beta
    return V,H


#Factorizes A as
#A * V[:,0:k] = V * H
#Where V spans a Krylov space of A
#and V is orthogonal in the supplied inner product `dot`
def arnoldi_general(A,v,k,dot):
    norm = lambda x: np.sqrt(dot(x,x))
    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        h=dot(V[:,0:j+1],w)
        f=w-V[:,0:j+1]@h
        s=dot(V[:,0:j+1],f)
        f=f-V[:,0:j+1]@s
        h=h+s
        beta=norm(f)
        H[0:j+1,j]=h
        H[j+1,j]=beta
        V[:,j+1]=f.flatten()/beta
    return V,H


def arnoldi_general_basis(H,v,z,beta):
    _,k=H.shape
    P=np.zeros((len(z),k+1))
    P[:,0]=1.0/beta
    for j in range(0,k):
        w=z*P[:,j]
        for i in range(0,j+1):
            w=w-H[i,j]*P[:,i]
        P[:,j+1]=w/H[j+1,j]
    return P

#  ||AVky - b||_w
#= ||VHy - b||_w
#= ||sqrt(W)*(VHy - b)||_2
#= ||Hy - e||_2

def gmres_general_step(A,b,k,dot):
    m=len(b)
    norm = lambda x: np.sqrt(dot(x,x))
    x=np.zeros((m,))
    r=b-A(x)
    V,H=arnoldi_general(A,r,k,dot)

    beta=norm(r)
    e=np.zeros(k+1)
    e[0]=beta
    y,_,_,_=la.lstsq(H,e)
    x=x+V[:,0:k]@y
    return x,y



def polynomial_step(A,b,x0,H,y,beta): 
    m=b.shape[0]
    r = b - A(x0)
    _,k=H.shape
    V=np.zeros((m,k+1))
    V[:,0]=r/beta
    for j in range(0,k):
        w=A(V[:,j])
        for i in range(0,j+1):
            w=w-H[i,j]*V[:,i]
        V[:,j+1]=w/H[j+1,j]
    return x0+V[:,0:k]@y,r - A(V[:,0:k]@y)








def arnoldi_basis(H,v,z):
    _,k=H.shape
    P=np.zeros((len(z),k+1),dtype=H.dtype)
    P[:,0]=1.0/np.linalg.norm(v)
    for j in range(0,k):
        w=z*P[:,j]
        for i in range(0,j+1):
            w=w-H[i,j]*P[:,i]
        P[:,j+1]=w/H[j+1,j]
    return P




# Solve: Ax=b
# min || (AVk)y - b|| (x=Vky)
# ||(AVk)y - b|| = || VHy - b ||
# ||Hy - V^T b||
# ||Hy - beta*e||

def gmres_step(A,b,z,k):
    m=len(b)
    x=np.zeros((m,))
    r=b-A(x)
    V,H=arnoldi(A,r,k)
    P=arnoldi_basis(H,r,z)

    beta=np.linalg.norm(r)
    e=np.zeros(k+1)
    e[0]=beta
    y,_,_,_=la.lstsq(H,e)
    x=x+V[:,0:k]@y
    return x,1-z*(P[:,0:k]@y)


#def solver_step(A,b,x0,F):
#    r = b - A(x0)
#    V,H = F(r)
#    AV = A(V)
#    e,res,rank,s = np.linalg.lstsq(AV,r)
#    return x0 + V@e
