### traffic flow PDE model

import numpy
from matplotlib import pyplot

# enumerated model parameters
V_max = [80.0, 136]             # maxmimum traffic speed, [km/hr]
rho0mult = [10, 20]             # traffic density IC multiplier
rho0_t = [10, 20]               # boundary condition on traffic density, [cars/km]

for j, (V_max, rho0mult, rho0_t) in enumerate(zip(V_max, rho0mult, rho0_t)):
    # model parameters
    L = 11.0                        # length of the road, [km]
    rho_max = 250.0                 # maximum traffic density, [cars/km]
    nx = 51                         # number of spacial grid points
    dt = .001                       # time step size, [hr]
    T = 0.1                         # total simulation time, [hr]
    nt = int(T/dt) + 1              # number of time steps
    t = numpy.linspace(0, T, nt)
    x = numpy.linspace(0, L, nx)
    dx = L / (nx - 1)               # spacial grid size, [m]

    # initial conditions
    rho0 = numpy.ones(nx)*rho0mult
    rho0[10:20] = 50
    rho_t = numpy.zeros((nx, nt))
    rho_t[:, 0] = rho0
    V = numpy.zeros((nx, nt))

    # boundary conditions
    rho_t[0, :] = rho0_t

    def F(r):
        return V_max*r*(1-r/rho_max)

    # run the calcuations
    for n in range(nt-1):
        rho_n = rho_t[:, n].copy()
        Fn = F(rho_n)
        rho_t[1:, n+1] = rho_n[1:] - dt/dx*(Fn[1:] - Fn[0:-1])
        rho_t[0 , n+1] = rho0_t

    # Traffic velocity, [m/s]
    V = V_max * (1 - rho_t/rho_max)*1000/(60*60)

    print('For Vmax = {:.1f} m/s, rho0mult = {} cars/km, and rho(0, t) = {} cars/km'.format(V_max, rho0mult, rho0_t))


    # Find the minimum velocity at t=0, [m/s]
    print('The minimum velocity at t = 0 is {:.2f} m/s'.format(min(V[:, 0])))

    # Find the average velocity at t=3 min
    t_min = t * 60
    N_t_3min = numpy.where(t_min == 3)[0][0]
    print('The average velocity at t = 3 min is {:.2f} m/s'.format(numpy.average(V[:, N_t_3min])))

    # Find the minimum velocity at t=6 min
    N_t_6min = numpy.where(t_min == 6)[0][0]
    print('The minimum velocity at t = 6 min is {:.2f} m/s'.format(min(V[:, N_t_6min])))
    print('')