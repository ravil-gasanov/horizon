import numpy as np

from rpy2.robjects import r
import anndata2ri
import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2ri
from rpy2 import robjects;
from rpy2.robjects.packages import importr
import re
from rpy2.robjects.conversion import localconverter

rpy2.robjects.numpy2ri.activate()

MARS = importr('earth');

class Mars:
    def fit_predict(self, X, y):
        return REarth(X, y)

# the following code is taken from Globe implementation by Mian et al.
# https://eda.rg.cispa.io/prj/globe/

def REarth(X,Y,M=1): # pragma: no cover
	row,col=X.shape

	try:
		rX=r.matrix(X.to_numpy(),ncol=col,byrow=False)
		rY=r.matrix(Y.to_numpy(),ncol=1,byrow=True)
	except AttributeError:
		rX=r.matrix(X,ncol=col,byrow=False)
		rY=r.matrix(Y,ncol=1,byrow=True)

	try:
		rearth=MARS.earth(x=rX,y=rY,degree=M)
	except:
		print("Singular fit encountered, retrying with Max Interactions=1");
		rearth=MARS.earth(x=rX,y=rY,degree=1)

	RSS_INDEX=0;
	DIRS_INDEX=5;
	CUTS_INDEX=6;
	SELECTED_INDEX=7;
	COEFF_INDEX=11;

	no_of_terms=np.size(rearth[SELECTED_INDEX]);

	#print('-------')
	#first we extract the hinges that were finally selected by MARS
	working_index=np.array(rearth[SELECTED_INDEX].flatten(),dtype=int)-1; 
	#print("WI: ",working_index)
	#print("Orig: ",rearth[SELECTED_INDEX].flatten())
	
	
	#next we check if these selected hinges contain all the variables that were present in X
	dir_rows=rearth[DIRS_INDEX][working_index,:]; 
	dirs=np.sum(np.abs(dir_rows),axis=0);	
	unused= (len(np.flatnonzero(dirs))+ 1) < X.shape[1]; 		#+1 is added to take into account the all 1's column, seems like MARS uses its own intercept term so our 1's col is set to zero always.
	#print('-------')
	#print("Dirs: ",dirs)
	#print("Unused: ",unused)

	#next we would like to know the number of terms in each hinge
	#we can do this by taking row sum of the selected Dirs
	interactions=[];
	for j in range(dir_rows.shape[0]):
		int_row=dir_rows[j,:];
		ints = np.sum(np.array(int_row!=0,dtype=int))
		interactions.append(ints);

	
	#print('-------')
	#print(interactions);
	#next we would like to record the coefficients
	#print('-------')	
	coeffs=[];
	cut_rows=rearth[CUTS_INDEX][working_index,:];
	for j in range(cut_rows.shape[0]):
		c_row=cut_rows[j,:];
		c_index=np.flatnonzero(c_row);
		for ci in c_index:
			coeffs.append(c_row[ci]);
	#print("Coeff: ",coeffs)
	
	reg_coeffs=rearth[COEFF_INDEX].reshape((-1,1));
	for j in range(reg_coeffs.shape[0]):
		coeffs.append(reg_coeffs[j,0]);
	#print("Coeff: ",coeffs)

	sse=rearth[RSS_INDEX][0]
	#print("sse: ",sse)
	
	return sse,[coeffs],np.array([no_of_terms]),interactions;
