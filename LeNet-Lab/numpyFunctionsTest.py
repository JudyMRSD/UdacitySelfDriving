# padding
# (array, pad_width, mode, **kwargs)
import numpy as np

a = np.array([[ 1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.]])

print (a)

print ("padding ")

# (padding before , padding after)
# (0, 1)  padding for the first dimension (row)
# (0, 3) padding for the second dimension (column)
# mode='constant': pad with constant value: Default is 0
b = np.pad(a, [(4, 1), (0, 3)], mode='constant')

print (b)