from math import sin, cos, log, ceil
import numpy

gridSize = numpy.array([0.04, 0.02, 0.01])
C_D = numpy.array([1.600, 1.500, 1.475])

r = gridSize[0]/gridSize[1]
p = log((C_D[0]-C_D[1])/(C_D[1]-C_D[2]))/log(r)
print(p)