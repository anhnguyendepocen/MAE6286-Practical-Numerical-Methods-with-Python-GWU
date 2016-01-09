import numpy as np


# Linspoace
N = 43
x = np.linspace(4, 23, N)
print(x[5])

# Array Dimensions
ones_array = np.ones( (5,17) )
zeros_array = np.zeros( ones_array.shape )

# Array Trigonometry
p = 7
r = np.array([11.2, 4.7, 6.6])
print(np.sin(p/r)**3)