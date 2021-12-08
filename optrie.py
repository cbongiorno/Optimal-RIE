import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def OptRIE(v,Sout,sorteig=False):
    '''
    v: in-sample eigenvectors (covariance)
    Sout: out-of-sample covariance matrix
    sorteig: Boolean, if True the eigenvalues are returns in increasing order
    
    return: Optimal eigenvalues
    '''
    
    N = v.shape[0]
    indx = np.triu_indices(N,0)

    ### Ax = b
    
    #Inverted matrix
    A = v[indx[0]]*v[indx[1]]
    A = np.column_stack((A,-np.identity(A.shape[0])))
    
    #Sum by rows of the inverted matrix
    A = np.row_stack((A,np.zeros((N,A.shape[1]))))
    for i in range(N):
        A[int(N*(N+1)/2)+i] = np.concatenate((np.zeros(N),(np.logical_or(indx[0]==i,indx[1]==i)).astype(float)))
    A = np.column_stack((A,np.zeros((A.shape[0],N))))
    A[N*(N+1)//2 : N*(N+1)//2+N, N+N*(N+1)//2 : 2*N+N*(N+1)//2] = -np.identity(N)
    
    # Normalizzation of the ptf wieghts
    A = np.row_stack((A,  np.concatenate((np.zeros(N+N*(N+1)//2),np.ones(N)))))

    b = np.zeros(A.shape[0])
    b[-1] = 1.

    
    #### Gx<=b
    
    #positive eigenvalues
    G = np.zeros((N,A.shape[1]))
    G[:N,:N] = -np.identity(N)
    
    if sorteig==True:
        #to have sorted eigenvalues
        G = np.row_stack((G,np.zeros((N-1,G.shape[1]))))
        G[N:,:N-1] = -np.identity(N)[:N-1,:N-1]
        G[N:,1:N]+=np.identity(N)[:N-1,:N-1]

    h = np.zeros(G.shape[0])

    ## x P x (minimize the variance of the ptf)
    P = np.zeros((A.shape[1],A.shape[1]))
    P[N+N*(N+1)//2:,N+N*(N+1)//2:] = Sout
    q = np.zeros(A.shape[1])

    out = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))

    x = np.array(out['x']).T[0]

    # return eigenvalues (there is a undetermined scale factor)
    lopt = 1/x[:N]
    return lopt/lopt.sum()
