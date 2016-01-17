import numpy
from matplotlib import pyplot
from matplotlib import animation

def u_initial(nx):
    """
    Defines the initial conditions
    
    u(x, 0) = 1, 0 <= x < x_max/2
            = 0, x_max/2 <= x <= x_max
    """
    
    u = numpy.zeros(nx)
    u[0:int(nx/2)] = 1.0
    return u

def computeF(u):
    """
    Computes the flux
    
    Parameters
    ----------
    
    u    : array of floats
            Velocity
    """
    return u**2 / 2

def maccormack(u, nt, dt, dx, eps):
    """
    Computes the solution with the MacCormack scheme
    
    Parameters
    ----------
    u    : array of floats
            velocity at current time step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    """
    
    un = numpy.zeros((nt,len(u)))
    un[:] = u.copy()
    ustar = u.copy()    

    for n in range(1,nt):
        F = computeF(u)
        
        ustar[1:-1] = u[1:-1] - dt/dx * (F[2:] - F[1:-1]) + eps*(u[2:] - 2*u[1:-1] + u[:-2])
        Fstar = computeF(ustar)
        un[n, 1:] = 0.5 * ( u[1:] + ustar[1:] - dt/dx * (Fstar[1:] - Fstar[:-1]) ) 
        u = un[n].copy()

    return un

nx = 81
nt = 70
dx = 4.0/(nx-1)
eps = 0.25

def animate(data):
    x = numpy.linspace(0,4,nx)
    y = data
    line.set_data(x,y)
    return line,

u = u_initial(nx)
sigma = .5
dt = sigma*dx

un = maccormack(u,nt,dt,dx,eps)

fig = pyplot.figure()
ax = pyplot.axes(xlim=(0,4),ylim=(-.5,2))
line, = ax.plot([],[],lw=2)

anim = animation.FuncAnimation(fig, animate, frames=un, interval=50)

pyplot.show()