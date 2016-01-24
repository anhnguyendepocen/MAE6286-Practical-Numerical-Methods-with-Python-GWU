import numpy
from matplotlib import pyplot
from matplotlib import animation


##### Reaction-Diffusion system, forward-time and central-difference in space

def laplacian(X, dh):
    """
    Compute the laplacian in 2D using central differences
    :param X: array X[y, x]
    :param dh: grid spacing, equal in x and y directions
    :return: array of the laplacian
    """

    Xip1_j = X[1:-1, 2:]
    Xi_j = X[1:-1, 1:-1]
    Xim1_j = X[1:-1, :-2]
    Xi_jp1 = X[2:, 1:-1]
    Xi_jm1 = X[:-2, 1:-1]

    return numpy.array([1./(dh**2)*(Xip1_j - 4.*Xi_j + Xim1_j + Xi_jp1 + Xi_jm1)])


def rhs_U(u, v, dh, Du, F):
    """Computes the forcing term for U

    Parameters
    ----------
    u  : array of floats
        vectors [u, v] at every point x, y
    dh: float
        grid spacing
    Du, F:   floats
        contants

    Returns
    -------
    F : array
        Array for U forcing term at every point x, y
    """

    Ui_j = u[1:-1, 1:-1]
    Vi_j = v[1:-1, 1:-1]

    return numpy.array([Du*laplacian(u, dh) - Ui_j*Vi_j**2 + F*(1-Ui_j)])

def rhs_V(u, v, dh, Dv, F, k):
    """Computes the forcing term for V

    Parameters
    ----------
    u  : array of floats
        vectors [u, v] at every point x, y
    dh: float
        grid spacing
    Dv, F, k:   floats
        contants

    Returns
    -------
    F : array
        Array for V forcing term at every point x, y
    """

    Ui_j = u[1:-1, 1:-1]
    Vi_j = v[1:-1, 1:-1]

    return numpy.array([Dv*laplacian(v, dh) + Ui_j*Vi_j**2 - (F+k)*Vi_j])

def ftcd(u, v, nt, dt, dh, Du, Dv, F, k):
    """
    Compute the solution using forward-time and central-difference in space
    :type u: numpy.array[2, ny, nx]
    :param u: solution array (t, 2, y, x)
    :param nt: number of time steps
    :param dt: time step size
    :param dh: gride spacing, equal in x, y
    :param Du: parameter
    :param Dv: parameter
    :param F: parameter
    :param k: parameter
    :return: array with time evolution of the solution array at all points [y, x]
    """

    # Array indexing example
    # A = [0, 1, 2, 3, 4, 5, 6]
    # A[:-1] = [0 1 2 3 4 5]
    # A[1:] = [1 2 3 4 5 6]
    # A[1:-1] = [1 2 3 4 5]
    # A[:-2] = [0 1 2 3 4]
    # A[2:] = [2 3 4 5 6]

    # copy the initial u array into our new array
    ut = numpy.zeros_like(u)
    vt = numpy.zeros_like(v)
    ut = u.copy()
    vt = v.copy()

    for t in range(1,nt):

        # compute forcing term
        forceU = rhs_U(u, v, dh, Du, F)
        forceV = rhs_V(u, v, dh, Dv, F, k)

        # compute u^(n+1), v^(n+1)
        ut[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*forceU
        vt[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*forceV

        # apply Neumann boundary conditions, qx = qy = 0
        ut[:, 0] = ut[:, 1]
        ut[:, -1] = ut[:, -2]
        ut[0, :] = ut[1, :]
        ut[-1, :] = ut[-2, :]

        vt[:, 0] = vt[:, 1]
        vt[:, -1] = vt[:, -2]
        vt[0, :] = vt[1, :]
        vt[-1, :] = vt[-2, :]


        # Neumann conditions at the corners

        u = ut.copy()
        v = vt.copy()

    return ut


# model parameters
n = 192
Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065 # Bacteria 1
dh = 5./(n-1)
dt = .9 * dh**2 / (4*max(Du,Dv))
T = 8000                           # Run time in seconds
nt = int(T/dt)
xDomain = (0, 5)
yDomain = (0, 5)

# Initial conditions
# initialize our results array with dimensions nt by 2 by ny by nx
U = numpy.zeros((n, n))
V = numpy.zeros_like(U)
uvinitial = numpy.load('./uvinitial.npz')
U = uvinitial['U']
V = uvinitial['V']

Uresult = ftcd(U, V, nt, dt, dh, Du, Dv, F, k)

# Results
result = Uresult[100, ::40]
print(numpy.around(result, decimals=4))
