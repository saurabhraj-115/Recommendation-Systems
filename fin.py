
from __future__ import print_function                                            #importing relevant tools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
from os import system
import math
from itertools import product as pr
# from sklearn.impute import SimpleImputer





def spa(X):
    non_zero = np.count_nonzero(X)
    total_val = np.product(X.shape)
    sparsity = (total_val - non_zero)
    return sparsity








system('clear')
A=np.genfromtxt("movie_rarting_test.csv",delimiter=',',skip_header=1)            #importing dataset
# A=pd.DataFrame(A)
D=np.genfromtxt("rsdata.csv",delimiter=',',skip_header=1)                        #importing dataset
# D=pd.
A= A[1:,1:]                                                                      #deleting the coulumn with user ID
D= D[1:,1:]                                                                      #deleting the coulumn with user ID
#

A=A[:D.shape[0],:]

B= np.zeros((D.shape[0],D.shape[0]))                                             #making a matrix for storing cosine_similarity
for i,j in pr(range(D.shape[0]) , range(D.shape[0])):                            #implementing cosine_similarity
        M=D[i,:].reshape(1,-1)
        N=D[j,:].reshape(1,-1)
        B[i,j]= cosine_similarity(M,N)


k=int(math.floor(A.shape[0]/3.0))
top=int(math.floor(k/3.0))

flag=0
c=0
q=3
C=np.zeros((A.shape[0],A.shape[1]))                                                                      #this parameter could be changed accordin to convinience
while flag==0 and q>-1:
    q-=1
    K= np.argpartition(B, np.argmin(B, axis=0))[:, -k-1:]                            #partitioning the datasets row-wise and finding the closest 'top'+1 users to each users
    K= K[:,int((q)*top):int((q+1)*top-1)]                                            #removing itself from top+1 users closest to itself
    for i,j in pr(range(A.shape[0]),range(A.shape[1])):
        # if A[i,j] == 0:
        su= 0.0
        ctr=0.0
        for l in range(int(top-1)):
            if A[K[i,l],j] > 0.0:
                flag=1
                c=1
                su+= A[K[i,l],j]
                ctr+=1.0
        if flag==1:
                C[i,j]=float(su/ctr)
                flag=0

    if c==1 and q==2 and A[i,j]==0:
        C[i,j]=0

# C= np.around(C)                                                                #rounding off the numpy array from float to nearest integer rating
# print (C)
# A=pd.DataFrame(A)
# C=pd.DataFrame(C)
C=  np.ceil(C)
# print (A)
# print (C)
# print (spa(C))
# np.savetxt('final_ratings.csv', C, delimiter=',',fmt='%1.18f')  #outputting to any file
nz=0
mae=0
rmse=0
for i,j in pr(range(A.shape[0]),range(A.shape[1])):
    if A[i,j]>0:
        nz+=1
        mae+=abs(A[i,j]-C[i,j])


mae=mae/nz
for i,j in pr(range(A.shape[0]),range(A.shape[1])):
    if A[i,j]>0:
        rmse=(C[i,j]-A[i,j]-mae)**2
rmse=np.sqrt(rmse/nz)
A=pd.DataFrame(A)
C=pd.DataFrame(C)
print (A)
print (C)
print (spa(C))
print ('Mae and RMSE are')
print (mae)
print (rmse)
