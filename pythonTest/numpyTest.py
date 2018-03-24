import numpy as np
a, ai, ac = np.unique(np.array([1, 3,3,3, 2, 3, 2, 2]), return_index=True,return_counts=True)
print (a,ai,ac)
# a =[1 2 3]  unique numbers in array a
# ai = [0 4 1]  index for where the unique numbers occured in the array
# 1, 3,3,3, 2, 3, 2, 2
# ^  ^      ^
# 0  1      4
# ac = [1 3 4] count of number 1,2,3 in array a

