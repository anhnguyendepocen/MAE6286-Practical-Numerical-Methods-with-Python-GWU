import numpy
from math import pi
from matplotlib import pyplot, cm
from laplace_helper import plot_3D

###  Solve stokes lid-driven cavity flow

def BCs(omega, psi, u_top_bndry, dx, dy):
    psiBC = psi.copy()
    omegaBC = omega.copy()
    psiBC[0, :] = 0.
    psiBC[:, 0] = 0.
    psiBC[:, -1] = 0.
    psiBC[-1, :] = 0.
    psiBC[:, 1] = psiBC[:, 0]
    psiBC[:, -2] = psiBC[:, -1]
    psiBC[1, :] = psiBC[0, :]
    psiBC[-1, :] = u_top_bndry * dy + psiBC[-2, :]
    omegaBC[-1, :] = -1/(2*dy**2)*(8*psiBC[-2, :] - psiBC[-3, :]) - 3*u_top_bndry/dy
    return omegaBC, psiBC

def L1norm(new, old):
    norm = numpy.sum(numpy.abs(new-old))
    return norm

def resid(u, b, dx, dy):
    nx, ny = u.shape
    result = numpy.zeros((ny,nx))
    result[1:-1, 1:-1] = b[1:-1, 1:-1]*dx**2*dy**2 - \
                         ( (u[1:-1, 2:] -2*u[1:-1, 1:-1] + u[1:-1, :-2])*dy**2 + \
                         (u[2:, 1:-1] -2*u[1:-1, 1:-1] + u[:-2, 1:-1])*dx**2 )
    return result

def Ad(u, dx, dy):
    nx, ny = u.shape
    result = numpy.zeros((ny,nx))
    result[1:-1, 1:-1] = (u[1:-1, 2:] -2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2 + \
                         (u[2:, 1:-1] -2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
    return result

def conjugate_gradient_2d(omega, psi, dx, dy, l1_target):
    '''Performs cg relaxation
    Assumes Dirichlet boundary conditions p=0

    Parameters:
    ----------
    p : 2D array of floats
        Initial guess
    b : 2D array of floats
        Source term
    dx: float
        Mesh spacing in x direction
    dy: float
        Mesh spacing in y direction
    l2_target: float
        exit criterion

    Returns:
    -------
    p: 2D array of float
        Distribution after relaxation
    '''
    ny, nx = omega.shape
    r_omega  = numpy.zeros((ny,nx)) # residual
    Ad_omega  = numpy.zeros((ny,nx)) # to store result of matrix multiplication
    r_psi  = numpy.zeros((ny,nx)) # residual
    Ad_psi  = numpy.zeros((ny,nx)) # to store result of matrix multiplication

    l1_norm_omega = 1
    l1_norm_psi = 1
    iterations = 0

    # Step-0 We compute the initial residual and
    # the first search direction is just this residual

    r_omega = resid(omega, numpy.zeros((ny,nx)), dx, dy)
    r_psi = resid(psi, -omega, dx, dy)

    d_omega = r_omega.copy()
    d_psi = r_psi.copy()
    rho_omega = numpy.sum(r_omega*r_omega)
    rho_psi = numpy.sum(r_psi*r_psi)

    Ad_omega = Ad(d_omega, dx, dy)
    Ad_psi = Ad(d_psi, dx, dy)

    sigma_omega = numpy.sum(d_omega*Ad_omega)
    sigma_psi = numpy.sum(d_psi*Ad_psi)

    # Iterations
    while l1_norm_omega > l1_target and l1_norm_psi > l1_target:

        omegak = omega.copy()
        psik = psi.copy()
        r_omegak = r_omega.copy()
        r_psik = r_psi.copy()
        d_omegak = d_omega.copy()
        d_psik = d_psi.copy()

        alpha_omega = rho_omega/sigma_omega
        alpha_psi = rho_psi/sigma_psi

        omega = omegak + alpha_omega*d_omegak
        psi = psik + alpha_psi*d_psik
        r_omega = r_omegak - alpha_omega*Ad_omega
        r_psi = r_psik - alpha_psi*Ad_psi

        rho_omegap1 = numpy.sum(r_omega * r_omega)
        rho_psip1 = numpy.sum(r_psi * r_psi)
        beta_omega = rho_omegap1 / rho_omega
        beta_psi = rho_psip1 / rho_psi
        rho_omega = rho_omegap1
        rho_psi = rho_psip1

        d_omega = r_omega + beta_omega*d_omegak
        d_psi = r_psi + beta_psi*d_psik

        Ad_omega = Ad(d_omega, dx, dy)
        Ad_psi = Ad(d_psi, dx, dy)
        sigma_omega = numpy.sum(d_omega*Ad_omega)
        sigma_psi = numpy.sum(d_psi*Ad_psi)

        # BCs
        omega, psi = BCs(omega.copy(), psi.copy(), u_top_bndry, dx, dy)

        l1_norm_omega = L1norm(omega, omegak)
        l1_norm_psi = L1norm(psi, psik)
        iterations += 1

    print('Number of CG iterations: {0:d}'.format(iterations))
    return omega, psi

# Parameters
nx = 41
ny = 41

l = 1.
h = 1.

dx = l/(nx-1)
dy = h/(ny-1)

l1_target = 1e-6

#Initial Conditions
omega_i = numpy.zeros((ny, nx))
psi_i = numpy.zeros((ny, nx))
u_top_bndry = 1.
omega_i, psi_i = BCs(omega_i.copy(), psi_i.copy(), u_top_bndry, dx, dy)

omega, psi = conjugate_gradient_2d(omega_i.copy(), psi_i.copy(), dx, dy, l1_target)

# Mesh
x  = numpy.linspace(0,1,nx)
y  = numpy.linspace(0,1,ny)
X,Y = numpy.meshgrid(x,y)

