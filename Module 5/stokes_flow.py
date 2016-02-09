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
                         ( (u[1:-1, 2:] -2*u[1:-1, 1:-1] + u[1:-1, :-2])*dy**2 +
                         (u[2:, 1:-1] -2*u[1:-1, 1:-1] + u[:-2, 1:-1])*dx**2 )
    return result

def Ad_fcn(u, dx, dy):
    nx, ny = u.shape
    result = numpy.zeros((ny,nx))
    result[1:-1, 1:-1] = (u[1:-1, 2:] -2*u[1:-1, 1:-1] + u[1:-1, :-2])*dx**2 + \
                         (u[2:, 1:-1] -2*u[1:-1, 1:-1] + u[:-2, 1:-1])*dy**2
    return result

def laplace2d(p, psi, dx, dy, l1_target, u_top_bndry):
    '''Iteratively solves the Laplace equation using the Jacobi method '''
    l1norm = 1
    pn = numpy.empty_like(p)
    while l1norm > l1_target:
        pn = p.copy()
        p[1:-1,1:-1] = 0.5/(dx**2 + dy**2) * \
                       ( (pn[1:-1,2:] + pn[1:-1, :-2])*dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1])*dx**2 )

        ##Neumann B.C. along x = L
        p[-1, :] = -.5/(dy**2)*(8*psi[-2,:]-psi[-3,:]) - 3*u_top_bndry/dy
        l2norm = L1norm(p, pn)
    return p

def conjugate_gradient_2d(p, b, dx, dy, l1_target, u_top_bndry):

    # p  = psi
    # b = -omega
    ny, nx = p.shape
    r  = numpy.zeros((ny,nx)) # residual
    Ad  = numpy.zeros((ny,nx)) # to store result of matrix multiplication

    l1_norm = 1
    iterations = 0

    # Step-0 We compute the initial residual and
    # the first search direction is just this residual

    r = resid(p, b, dx, dy)
    d = r.copy()
    rho = numpy.sum(r*r)
    Ad = Ad_fcn(d, dx, dy)
    sigma = numpy.sum(d*Ad)

    # Iterations
    while l1_norm > l1_target:

        pk = p.copy()
        rk = r.copy()
        dk = d.copy()
        bk = b.copy()

        b = laplace2d(bk, pk, dx, dy, l1_target, u_top_bndry)

        alpha = rho/sigma

        p = pk + alpha*dk
        r = rk - alpha*Ad

        rhop1 = numpy.sum(r*r)
        beta = rhop1 / rho
        rho = rhop1

        d = r + beta*dk
        Ad = Ad_fcn(d, dx, dy)
        sigma = numpy.sum(d*Ad)

        # BCs are automatically enforced

        l1_norm = L1norm(pk, p)
        iterations += 1

    print('Number of CG iterations: {0:d}'.format(iterations))
    return p, b

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
psi, omega = conjugate_gradient_2d(omega_i.copy(), -psi_i.copy(), dx, dy, l1_target, u_top_bndry)

print('maximum value of abs(psi): {0:.4f}'.format(numpy.amax(numpy.absolute(psi))))
print('maximum value of abs(omega): {0:.4f}'.format(numpy.amax(numpy.absolute(omega))))

# Mesh
x  = numpy.linspace(0,1,nx)
y  = numpy.linspace(0,1,ny)

plot_3D(x, y, psi)

