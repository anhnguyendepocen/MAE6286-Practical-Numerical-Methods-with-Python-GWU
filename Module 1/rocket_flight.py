from math import sin, cos, log, ceil
import numpy
from matplotlib import pyplot

# model parameters:
m_s = 50.0                      # mass of the rocket shell in kg
g = 9.81                        # gravity in m s^{-2}
rho = 1.091                     # average air density in kg m^{-3}
r = 0.5                         # cross-sectional radius of rocket in m
A = numpy.pi * r**2             # maximum cross-sectional area of the rocket in m^2
v_e = 325.0                     # exhaust speed in m s^{-1}
C_D = 0.15                      # drag coefficient --- or D/L if C_L=1

### set initial conditions ###
m_p0 = 100.0                    # initial mass of the rocket propellent in kg
v0 = 0.0                        # start at rest
h0 = 0.0                        # initial altitude

### solution parameters
dt = 0.1                        # time step in s
T = 40                          # final time
N = int(T/dt) + 1               # number of time steps
t = numpy.linspace(0, T, N)     # time discretization

# Propellent burn rate in kg s^{-1}
mp_dot = numpy.zeros(N)
n5 = int(5/dt)
mp_dot[:n5] = 20.0

# Compute m_p as a function of time
m_p = numpy.zeros(N)
for i in range(N):
    if i < n5:
        integral = mp_dot[i]*dt*i
    else:
        integral = mp_dot[0]*n5*dt
    m_p[i] = m_p0 - integral

def f(u, t):
    """Returns the right-hand side of the system of equations.

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """

    h = u[0]
    v = u[1]
    return numpy.array([v, -g + (mp_dot[t] * v_e -0.5*rho*v*numpy.abs(v)*A*C_D)/(m_s + m_p[t])])


def euler_step(u, f, dt, t):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f(u, t)

# initialize the array containing the solution
u = numpy.empty((N, 2))
u[0] = numpy.array([h0, v0])        # fill 1st element with initial values

# time loop - Euler method
for n in range(N-1):
    u[n+1] = euler_step(u[n],f,dt, n)


# print mass of rocket propellant at t = 3.2s
print('The mass of the rocket propellant is {:.3f} kg'.format(m_p[ int(3.2/dt) ]))

# print the maximum speed of the rocket in m s^{-1}
v_max = numpy.amax( u[:, 1])
print('The maximum speed of the rocket is {:.2f} m/s'.format(v_max))

# Find the time of the maximum speed in s
N_vmax = numpy.where(u[:, 1] == v_max)[0][0]
t_vmax = N_vmax * dt
print('The time when the maximum speed occurs is {:.2f} s'.format(t_vmax))

# Altitude in meters at t=t_vmax
print('The altitude at the time of the maximum speed is {:.2f} s'.format(u[:, 0][N_vmax]))

print('')

# The maximum altitude in m
h_max = numpy.amax( u[:, 0])
print('The maximum altitude of the flight is {:.2f} m'.format(h_max))

# Find the time of the maximum altitude in s
N_hmax = numpy.where(u[:, 0] == h_max)[0][0]
t_hmax = N_hmax * dt
print('The time when the maximum altitude occurs is {:.2f} s'.format(t_hmax))

# Find the time when the rocket hits the ground in s
print('')
N_heq0= numpy.where(u[:, 0] <0.0)[0][0]
t_heq0 = N_heq0 * dt
print('The time when the rocket impacts the ground is {:.2f} s'.format(t_heq0))

# print the velocity at impact of the rocket
print('The velocity at impact is {:.2f} m/s'.format(u[:, 1][N_heq0]))



# plot h
#pyplot.figure(figsize=(10,4))   #set plot size
#pyplot.grid(True)
# #pyplot.ylim(-5,105)             #y-axis plot limits
#pyplot.tick_params(axis='both', labelsize=14) #increase font size for ticks
#pyplot.xlabel('t', fontsize=14) #x label
#pyplot.ylabel('z', fontsize=14) #y label
#pyplot.plot(t, u[:, 0], 'k-');
#pyplot.show()

