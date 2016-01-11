import numpy
import sympy
from sympy.utilities.lambdify import lambdify

x = sympy.symbols('x')

phi = (sympy.cos(x)**2 * sympy.sin(x)**3)/(4* x**5 * sympy.exp(x))
phiprime = phi.diff(x)      # dphi/dx

dphidx = lambdify((x), phiprime)

print("The value of dphi/dx at x=2.2 is{}.".format(dphidx(2.2)))


#test enumerate
x1 = [1, 2]
x2 = [6, 9]
x3 = [7, 13]

for i, (a1, a2, a3) in enumerate(zip(x1, x2, x3)):
    print(i, a1, a2, a3)