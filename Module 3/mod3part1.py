import numpy
from matplotlib import pyplot
from matplotlib import animation
import sympy

##### Calculating constants
ustar = 1.5
umax = 2.0
rhomax = 15.0


u_max, u_star, rho_max, rho_star, A, B = sympy.symbols('u_max u_star rho_max rho_star A B')
eq1 = sympy.Eq( 0, u_max*rho_max*(1 - A*rho_max-B*rho_max**2) )
eq2 = sympy.Eq( 0, u_max*(1 - 2*A*rho_star-3*B*rho_star**2) )
eq3 = sympy.Eq( u_star, u_max*(1 - A*rho_star - B*rho_star**2) )

eq4 = sympy.Eq(eq2.lhs - 3*eq3.lhs, eq2.rhs - 3*eq3.rhs)

# Solve the equations
rho_sol = sympy.solve(eq4,rho_star)[0]
B_sol = sympy.solve(eq1,B)[0]
quadA = eq2.subs([(rho_star, rho_sol), (B,B_sol)])
A_sol = sympy.solve(quadA, A)

aval = A_sol[0].evalf(subs={u_star: ustar, u_max: umax, rho_max: rhomax})
print(aval)
bval = B_sol.evalf(subs={rho_max: rhomax, A:aval})
print(bval)

##### Maximum Density Limit
Fprime = eq1.rhs.diff(rho_max)
eq5 = sympy.Eq(0, Fprime)
rho_pos_wave = sympy.solve(eq5, rho_max)[0]
print(rho_pos_wave)
rhoval = rho_pos_wave.evalf(subs={A:aval, B:bval})
print(rhoval)