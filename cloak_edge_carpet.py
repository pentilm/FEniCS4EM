# Edge element for metamaterial carpet cloaking via frequency Maxwell's equation.
# For the physical model, please see:
# Li, J., Huang, Y., Yang, W. and Wood, A., 2014. Mathematical analysis and time-domain finite element simulation of carpet cloak. SIAM Journal on Applied Mathematics, 74(4), pp.1136-1151.
# Author: Zhiwei Fang
# Copyright reserved

from __future__ import print_function
from dolfin import *
from mshr import *
from math import sqrt
import sympy as syp
from dolfin_adjoint import *
from Geometry import *

# physical parameters
cc = 299792458.0	# Speed of light in vacuum
mu = 4.0*pi*1.0e-7
eps = 1.0/(cc*cc*mu)
w = cc*11			# frequency of the wave
k0 = (w/cc)**2	# wave number
wf = w/cc      # frequency of the wave of BC
xa = -1.0
ya = 0.0
xb = 1.0
yb = 2.0
H1 = 0.5
H2 = 1.0
Hd = 0.5
sint = H2/sqrt(H2**2 + Hd**2)	# angles used to defined the wave source
cost = Hd/sqrt(H2**2 + Hd**2)
# computational parameters
Nxy = 80	# spatial partition number
beta1 = beta2 = 1e-8		# regularization parameter

# piecewise defined material
tol_cor = -1e-8	# to avoid partition problem at the intersection of conductor and metamaterial, should add a tolerance
vet0 = [[-Hd,0], [Hd,0], [0,H1]]	# vertexes for conductor
vet1 = [[-Hd-tol_cor,0], [Hd+tol_cor,0], [0,H2]]	# vertexes for metamaterial
class Omega_1(SubDomain):	# the metamaterial domain
	def inside(self, x, on_boundary):
		return point_inside_polygon(x[0], x[1], vet1)
vet2 = [[-Hd+tol_cor,0], [Hd-tol_cor,0], [0,H2]]	# vertexes for metamaterial
class Omega_2(SubDomain):	# the vacuum domain
	def inside(self, x, on_boundary):
		return not point_inside_polygon(x[0], x[1], vet2)

domain = Rectangle(Point(xa, ya), Point(xb, yb)) - Polygon([Point(vet0[i][0], vet0[i][1]) for i in range(3)])
mesh = generate_mesh(domain, Nxy)


refine_domain = Omega_1()
refine_markers = MeshFunction("bool", mesh, mesh.topology().dim())
refine_markers.set_all(False)
refine_domain.mark(refine_markers, True)
mesh = refine(mesh, refine_markers)
##test mesh
# import matplotlib.pyplot as plt
# plt.figure()
# plot(mesh)
# plt.show()

# define function space for state and control
Vs = FunctionSpace(mesh, FiniteElement("N2curl", mesh.ufl_cell(), 1))	# function space for state function
Vc = FunctionSpace(mesh, "DG", 0)	# function space for control function

materials = MeshFunction("size_t", mesh, mesh.topology().dim())
materials.set_all(2)
subdomain_1 = Omega_1()
subdomain_1.mark(materials, 1)
dx = Measure('dx', domain=mesh, subdomain_data=materials)


# Source
# Define variables for symbolic computation
x, y = syp.symbols('x[0], x[1]')
Am = 1.0	# amplitude
Amc = 10	# length of the source wave
H_l = Am*syp.cos(wf*(-sint*x+cost*y))/syp.cosh(Amc*(cost*x+sint*y))
H_r = Am*syp.cos(wf*(sint*x+cost*y))/syp.cosh(Amc*(-cost*x+sint*y))
Ew_lx = syp.diff(H_l, y)/wf
Ew_rx = syp.diff(H_r, y)/wf
Ew_ly = -syp.diff(H_l, x)/wf
Ew_ry = -syp.diff(H_r, x)/wf
f_lx = (-syp.diff(H_l, x,2, y) - syp.diff(H_l, y,3) - k0*syp.diff(H_l, y))/wf
f_rx = (-syp.diff(H_r, x,2, y) - syp.diff(H_r, y,3) - k0*syp.diff(H_r, y))/wf
f_ly = (syp.diff(H_l, x, y,2) + syp.diff(H_l, x,3) + k0*syp.diff(H_l, x))/wf
f_ry = (syp.diff(H_r, x, y,2) + syp.diff(H_r, x,3) + k0*syp.diff(H_r, x))/wf
vars = [H_l, H_r, Ew_lx, Ew_rx, Ew_ly, Ew_ry, f_lx, f_rx, f_ly, f_ry]
vars = [syp.printing.ccode(var) for var in vars]
vars = [var.replace('M_PI', 'pi') for var in vars]
[H_l, H_r, Ew_lx, Ew_rx, Ew_ly, Ew_ry, f_lx, f_rx, f_ly, f_ry] = vars

Ewx_str = 'x[0]>=0.0 ? ' + Ew_rx + ' : ' + Ew_lx
Ewy_str = 'x[0]>=0.0 ? ' + Ew_ry + ' : ' + Ew_ly
fx_str = 'x[0]>=0.0 ? ' + f_rx + ' : ' + f_lx
fy_str = 'x[0]>=0.0 ? ' + f_ry + ' : ' + f_ly
Ew = Expression((Ewx_str, Ewy_str), element=Vs.ufl_element(), domain=mesh)
f = Expression((fx_str, fy_str), element=Vs.ufl_element(), domain=mesh)
# H_str = 'x[0]>=0.0 ? ' + H_r + ' : ' + H_l
# H = Expression(H_str, element=Vc.ufl_element(), domain=mesh)
## plot the source
# import matplotlib.pyplot as plt
# plt.figure()
# plot(interpolate(H, Vc),mesh)
# plt.show()

# boundary conditions
def bnd_in(x, on_boundary):
    return on_boundary and (x[0]<=Hd+tol_cor and x[0]>=-Hd-tol_cor and x[1]<H2)
bc1 = DirichletBC(Vs, Constant((0.0, 0.0)), bnd_in)
def bnd_out(x, on_boundary):
    return on_boundary and (not (x[0]<=Hd-tol_cor and x[0]>=-Hd+tol_cor and x[1]<H2))
bc2 = DirichletBC(Vs, Ew, bnd_out)
bcs = [bc1, bc2]

# given a control m, define the Helmholtz solver
nx = Constant((1.0, 0.0))
ny = Constant((0.0, 1.0))
def forward(epsr1, epsr2, epsr3, mur):
    # Solve the forward problem for a given material distribution
    E_sub = TrialFunction(Vs)
    v_sub = TestFunction(Vs)
	# continuous part
    F_sub = inner(mur*curl(E_sub), curl(v_sub))*dx(1) \
    - k0*((dot(E_sub,nx)*epsr1+dot(E_sub,ny)*epsr2)*dot(v_sub,nx)+(dot(E_sub,nx)*epsr2+dot(E_sub,ny)*epsr3)*dot(v_sub,ny))*dx(1) \
    + inner(curl(E_sub), curl(v_sub))*dx(2) - inner(k0*E_sub, v_sub)*dx(2) \
    - inner(f, v_sub)*dx(2)
    # dE_sub = TrialFunction(Vs)
    # dF_sub = derivative(F_sub, E_sub, dE_sub)
    E_sub = Function(Vs)
    solve(lhs(F_sub) == rhs(F_sub), E_sub, bcs)
    # solve(F_sub == 0, E_sub, bcs)
    return E_sub
epsr1e = Constant(H2/(H2-H1))
epsr2e = Expression("-H1*H2/(H2-H1)/Hd*(x[0]>=0 ? 1.0 : -1.0)",H1=H1, H2=H2, Hd=Hd, degree=0, domain=mesh)
epsr3e = Constant((H2-H1)/H2 + H2/(H2-H1)*(H1/Hd)**2)
mure = Constant((H2-H1)/H2)

# epsr1e = Constant(1.0)
# epsr2e = Constant(0.0)
# epsr3e = Constant(1.0)
# mure = Constant(1.0)

# epsr1e = Constant((H2/(H2-H1))**2)
# epsr2e = Expression("-H1*H2/(H2-H1)/Hd*(x[0]>=0 ? 1.0 : -1.0)*(H2/(H2-H1))",H1=H1, H2=H2, Hd=Hd, degree=0, domain=mesh)
# epsr3e = Constant(1.0 + (H2/(H2-H1)*H1/Hd)**2)
# mure = Constant(1.0)

epsr1 = interpolate(epsr1e, Vc)
epsr2 = interpolate(epsr2e, Vc)
epsr3 = interpolate(epsr3e, Vc)
mur = interpolate(mure, Vc)
E = forward(epsr1, epsr2, epsr3, mur)

# set the cost functional
Ed = interpolate(Ew, Vs)
J = assemble(0.5*inner(E - Ew, E - Ew)*dx(2) + 0.5*beta1*(epsr1**2+epsr2**2+epsr3**2)*dx(1) + 0.5*beta2*mur**2*dx(1))
dE = project(Ew-E, Vs)

print('objective functional value is: %f' % J)
file = File("cloak/cloak.pvd")
file << project(curl(E), FunctionSpace(mesh, "CG", 1))
file << project(curl(Ew), FunctionSpace(mesh, "CG", 1))
file << project(dot(E,nx), FunctionSpace(mesh, "CG", 1))
file << project(dot(E,ny), FunctionSpace(mesh, "CG", 1))
# file << epsr1
# file << epsr2
# file << epsr3
# file << mur
