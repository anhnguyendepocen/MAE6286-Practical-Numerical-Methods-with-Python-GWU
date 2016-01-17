import numpy
from matplotlib import pyplot
from matplotlib import animation


##### Sod's Shcok Tube test using Richtmyer method

def computeF(u, gamma):
    """Computes the flux

    Parameters
    ----------
    u  : array of floats
        vectors [rho, rho*u, rho*e_T] at every point x
    gamma: float
        specific heat ratio

    Returns
    -------
    F : array
        Array with flux vector at every point x
    """

    u1 = u[0, :]
    u2 = u[1, :]
    u3 = u[2, :]

    return numpy.array([u2,
                        u2**2/u1 + (gamma - 1.)*(u3 - 0.5*u2**2/u1),
                        u2/u1*(u3 + (gamma - 1.)*(u3 - 0.5*u2**2/u1))])

def u_initial(nx, gamma):
    """
    Defines the initial conditions
    IC = [rho, u, p]
    IC_L = [1 kg/m**3, 0 m/s, 100kN/m**2] for 10 <= x < 0
    IC_R = [0.125 kg/m**3, 0 m/s, 10 kN/m**2] for 0 <= x <= 10
        """
    IC_L = numpy.array([1., 0., 100000.])
    IC_R = numpy.array([0.125, 0., 10000.])

    def calc_ICs(IC, gamma):
        u1 = IC[0]
        u2 = IC[0] * IC[1]
        return numpy.array([u1, u2, IC[2]/(gamma - 1.) + 0.5 * u2**2 / u1])


    IC_L = calc_ICs(IC_L, gamma)
    IC_R = calc_ICs(IC_R, gamma)


    u = numpy.zeros((3, nx))

    u[0, 0:int(nx/2)] = IC_L[0]
    u[1, 0:int(nx/2)] = IC_L[1]
    u[2, 0:int(nx/2)] = IC_L[2]
    u[0, int(nx/2):] = IC_R[0]
    u[1, int(nx/2):] = IC_R[1]
    u[2, int(nx/2):] = IC_R[2]
    return u


def richtmyer(u, nt, dt, dx, gamma):
    """
    Computes the solution with the Richtmyer scheme

    Parameters
    ----------
    u    : array of floats
            vectors [rho, rho*u, rho*e_T]
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing

    Returns
    -------
    un : array of vectors of floats
    un = [rho, rho*u, rho*e_T] after nt time steps at every point x
    """

    # Array indexing example
    # A = [0, 1, 2, 3, 4, 5, 6]
    # A[:-1] = [0 1 2 3 4 5]
    # A[1:] = [1 2 3 4 5 6]
    # A[1:-1] = [1 2 3 4 5]
    # A[:-2] = [0 1 2 3 4]
    # A[2:] = [2 3 4 5 6]


    # initialize our results array with dimensions nt by 3 by nx
    un = numpy.zeros((nt, 3, nx))
    # copy the initial u array into our new array
    un[0, :, :] = u.copy()

    for n in range(1,nt):

        # step 1: compute u_star and fluxes from u_star
        #print(u)
        flux = computeF(u, gamma)
        ustar = 0.5 * (u[:, 1:] + u[:, :-1]) - \
                0.5 * dt/dx * (flux[:, 1:] - flux[:, :-1])
        fstar = computeF(ustar, gamma)

        #step 2: corrector
        un[n, :, 1:-1] = u[:, 1:-1] - dt/dx * (fstar[:, 1:] - fstar[:, :-1])
        un[n, :, 0] = un[n, :, 1]
        un[n, :, -1] = un[n, :, -2]
        u[:, :] = un[n].copy()

    return un


# model parameters
nx = 81
dx = 0.25
dt = 0.0002
gamma = 1.4                         # specific heat ratio for air
T = 0.012                           # Run time in seconds
nt = int(T/dt) + 1
xDomain = (-10, 10)
x = numpy.linspace(xDomain[0], xDomain[1], nx)
t = numpy.linspace(0, T, nt)

def animate(data):
    y = data
    line1.set_data(x,y)
    return line1,


# Initial conditions
u = u_initial(nx, gamma)
un = richtmyer(u, nt, dt, dx, gamma)

# Results
rho = numpy.zeros((nt, nx))
p = numpy.zeros((nt, nx))
U = numpy.zeros((nt, nx))

u1 = un[:, 0, :].copy()
u2 = un[:, 1, :].copy()
u3 = un[:, 2, :].copy()

rho = u1
U = u2/rho
p = (gamma - 1)*(u3 - 0.5*(u2**2/u1))

# Print results
N_v = numpy.where(x == 2.5)[0][0]
N_t = numpy.where(t == 0.01)[0][0]
print('The velocity U(x=2.5 m, t = 0.01 s) is {:.2f} m/s'.format(U[N_t, N_v]))
print('The pressure p(x=2.5 m, t = 0.01 s) is {:.2f} N/m**2'.format(p[N_t, N_v]))
print('The density rho(x=2.5 m, t = 0.01 s) is {:.2f} kg/m**3'.format(rho[N_t, N_v]))

# fig0 = pyplot.figure()
# ax0 = pyplot.axes(xlim=(xDomain[0], xDomain[1]),ylim=(numpy.amin(rho),numpy.amax(rho)))
# line1, = ax0.plot([],[],lw=2)
#
# anim = animation.FuncAnimation(fig0, animate, frames=rho, interval=50)

# Plot all three results at the desired time

pyplot.figure()
pyplot.subplot(311)
pyplot.title('Time = {:.2f} s'.format(N_t*dt))
pyplot.ylabel('Velocity (m/s)')
pyplot.grid(True)
pyplot.plot(x, U[N_t, :])

pyplot.subplot(312)
pyplot.ylabel('pressure (N/m**2)')
pyplot.grid(True)
pyplot.plot(x, p[N_t, :])

pyplot.subplot(313)
pyplot.ylabel('density (kg/m**3)')
pyplot.grid(True)
pyplot.plot(x, rho[N_t, :])

pyplot.show()