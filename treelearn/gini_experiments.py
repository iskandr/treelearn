import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

def score(L0, L1, R0, R1):
  NL = L0 + L1
  NR = R0 + R1
  N = NL + NR
  N = float(N)
  WL = NL / N
  WR = NR / N
  left = (1.0 / NL **2) * (L0 ** 2 + L1 ** 2)
  right = (1.0 / NR **2) * (R0 ** 2 + R1 ** 2)
  return WL * left + WR * right

N = 100
import pylab
import matplotlib
import matplotlib.pyplot as plt


def scatter():
  scores = np.ones((N,N,N)) * -1 

  X = []
  Y = []
  Z = []
  for idx in np.ndindex((N,N,N)):
    L0, L1, R0 = idx
    X.append(L0)
    Y.append(L1)
    Z.append(R0)
    R1 = N - (L0 + L1 + R0)
    if R1 >= 0 and (L0 + L1) > 0 and (R0 + R1) > 0:
        scores[idx] = score(L0, L1, R0, R1)

  X = np.array(X)
  Y = np.array(Y)
  Z = np.array(Z)
  C = np.ravel(scores)

  valid = C >= 0
  X = X[valid]
  Y = Y[valid]
  Z = Z[valid]
  C = C[valid] 

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter3D(X,Y,Z,c=C)
  ax.plot_surface(X,Y,Z,c=C)

import parakeet 
import numba 
#@numba.autojit 

@parakeet.jit
def _ksteps(max_k):
  changes = np.zeros(max_k)
  for L0 in xrange(N):
    for L1 in xrange(N):
      for R0 in xrange(N):
	    R1 = N - (L0 + L1 + R0)
	    if R1 >= 0 and (L0 + L1) > 0 and (R0 + R1) > 0:
	      curr_score = score(L0, L1, R0, R1)
	      for d_l0 in xrange(-max_k, max_k):
		l0 = L0 + d_l0 
		r0 = R0 - d_l0
		if l0 >= 0 and l0 <= N and r0 >= 0 and r0 <= N:
		  for d_l1 in xrange(-max_k, max_k):
		    l1 = L1 + d_l1
		    r1 = R1 - d_l1
		    if l1 >= 0 and l1 <= N and l1 >= 0 and l1 <= N:
		      nl = l0 + l1 
		      nr = r0 + r1
		      if nl > 0 and r0 >= 0 and r1 >= 0 and nr > 0 and (nl + nr) == N:
		        k = abs(d_l0) + abs(d_l1)
		        if k < max_k:	
		          new_score = score(l0, l1, r0, r1)
			  #print l0, l1, r0, r1, curr_score, new_score             
		          change_in_score = curr_score - new_score 
		          if change_in_score < changes[k]:
		            changes[k] = change_in_score 
  return changes 

def ksteps(max_k=50):
    changes = _ksteps(max_k)
    pylab.plot(changes)
  
#scatter()
ksteps()
pylab.show()


