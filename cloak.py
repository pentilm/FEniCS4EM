# Edge element for metamaterial cloaking via frequency Maxwell's equation.
# For the physical model, please see:
# Li, J. and Huang, Y., 2012. Mathematical simulation of cloaking metamaterial structures. Advances in Applied Mathematics and Mechanics, 4(1), pp.93-101.
# Author: Zhiwei Fang
# Copyright reserved

from __future__ import print_function
from dolfin import *
from mshr import *
from dolfin_adjoint import *

# physical parameters
cc = 299792458.0	# Speed of light in vacuum
mu = 4.0*pi*1.0e-7
eps = 1.0/(cc*cc*mu)
w = cc*11			# frequency of the wave
k0 = (w/cc)**2	# wave number
wf = w/cc      # frequency of the wave of BC
Rin = 0.3	# inside radius
Rout = 0.6	# outside radius
xa = ya = -2.0
xb = yb =2.0
xc = yc = 0.0	# center of conductor & matematerial
# computational parameters
Nxy = 100	# spatial partition number
Nm = 5	# number of metamaterial layers
gamma1 = gamma2 = 100	# penalty of the gradient
beta1 = beta2 = 1e-8		# regularization parameter

# piecewise defined material
tol = 1E-14			# tolerance for definition of domain
class Omega_1(SubDomain):	# the annular domain (metamaterial)
	def inside(self, x, on_boundary):
		val = (x[0]-xc)**2+(x[1]-yc)**2
		return val <= Rout**2 + tol
class Omega_2(SubDomain):	# the vacuum domain
	def inside(self, x, on_boundary):
		val = (x[0]-xc)**2+(x[1]-yc)**2
		return val>= Rout**2 - tol

domain = Rectangle(Point(xa, ya), Point(xb, yb)) - Circle(Point(xc, yc), Rin)
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
Vs = FunctionSpace(mesh, FiniteElement("N1curl", mesh.ufl_cell(), 1))	# function space for state function
# Vs = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 1))
Vc = FunctionSpace(mesh, "DG", 0)	# function space for control function

materials = MeshFunction("size_t", mesh, mesh.topology().dim())
materials.set_all(2)
subdomain_1 = Omega_1()
subdomain_1.mark(materials, 1)
dx = Measure('dx', domain=mesh, subdomain_data=materials)

# Source
Ew = Expression(("0.0","-cos(wf*x[0])"), wf = wf, element=Vs.ufl_element(), domain=mesh)
## plot the source
# import matplotlib.pyplot as plt
# plt.figure()
# plot(interpolate(Ew, Vs))
# plt.show()
f = Expression(("0.0","-(wf*wf-k0)*cos(wf*x[0])"), wf = wf, k0 = k0, element=Vs.ufl_element(), domain=mesh)

# boundary conditions
Tol = 1e-2
def bnd_in(x, on_boundary):
    return on_boundary and (x[0]-xc)**2+(x[1]-yc)**2<Rin**2+Tol
bc1 = DirichletBC(Vs, Constant((0.0, 0.0)), bnd_in)
def bnd_out(x, on_boundary):
    return on_boundary and (x[0]-xc)**2+(x[1]-yc)**2>Rin**2+Tol*10
bc2 = DirichletBC(Vs, Ew, bnd_out)
bcs = [bc1, bc2]

# given a control m, define the Helmholtz solver
nx = Constant((1.0, 0.0))
ny = Constant((0.0, 1.0))
# nml = FacetNormal(mesh)
# h = CellDiameter(mesh)
# h_avg = (h('+') + h('-'))/2
def forward(epsr1, epsr2, epsr3, mur):
    # Solve the forward problem for a given material distribution
    E_sub = Function(Vs)
    v_sub = TestFunction(Vs)
    F_sub = inner(mur*curl(E_sub), curl(v_sub))*dx(1) \
    - k0*((dot(E_sub,nx)*epsr1+dot(E_sub,ny)*epsr2)*dot(v_sub,nx)+(dot(E_sub,nx)*epsr2+dot(E_sub,ny)*epsr3)*dot(v_sub,ny))*dx(1) \
    + inner(curl(E_sub), curl(v_sub))*dx(2) - inner(k0*E_sub, v_sub)*dx(2) \
    - inner(f, v_sub)*dx(2)
    dE_sub = TrialFunction(Vs)
    dF_sub = derivative(F_sub, E_sub, dE_sub)
    solve(F_sub == 0, E_sub, bcs, J = dF_sub)
    # solve(F_sub == 0, E_sub, bcs)
    return E_sub
epsr1e = Expression("((1-R1/R2)*(1-R1/R2)+(1+2*(1-R1/R2)*(1-R1/R2)*R1/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))*x[1]*x[1]/(x[1]*x[1]+x[0]*x[0]))/((1-R1/R2)*(1-R1/R2)*sqrt(x[0]*x[0]+x[1]*x[1])/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))",R1=Rin,R2=Rout,degree=3,domain=mesh)
epsr2e = Expression("-(1+2*(1-R1/R2)*(1-R1/R2)*R1/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))*x[0]*x[1]/(x[0]*x[0]+x[1]*x[1])/((1-R1/R2)*(1-R1/R2)*sqrt(x[0]*x[0]+x[1]*x[1])/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))",R1=Rin,R2=Rout,degree=3,domain=mesh)
epsr3e = Expression("((1-R1/R2)*(1-R1/R2)+(1+2*(1-R1/R2)*(1-R1/R2)*R1/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))*x[0]*x[0]/(x[1]*x[1]+x[0]*x[0]))/((1-R1/R2)*(1-R1/R2)*sqrt(x[0]*x[0]+x[1]*x[1])/(sqrt(x[0]*x[0]+x[1]*x[1])-R1))",R1=Rin,R2=Rout,degree=3,domain=mesh)
mure = Expression("(1-R1/R2)*(1-R1/R2)*sqrt(x[0]*x[0]+x[1]*x[1])/(sqrt(x[0]*x[0]+x[1]*x[1])-R1)",R1=Rin,R2=Rout,degree=3,domain=mesh)

epsr1 = interpolate(epsr1e, Vc)
epsr2 = interpolate(epsr2e, Vc)
epsr3 = interpolate(epsr3e, Vc)
mur = interpolate(mure, Vc)
E = forward(epsr1, epsr2, epsr3, mur)

# set the cost functional
Ed = interpolate(Ew, Vs)
J = assemble(0.5*inner(E - Ew, E - Ew)*dx(2) + 0.5*beta1*(epsr1**2+epsr2**2+epsr3**2)*dx(1) + 0.5*beta2*mur**2*dx(1))
dE = project(E-Ew, Vs)

print('objective functional value is: %f' % J)
file = File("cloak/cloak.pvd")
file << project(curl(E), FunctionSpace(mesh, "CG", 1))
file << project(curl(dE), FunctionSpace(mesh, "CG", 1))
file << epsr1
file << epsr2
file << epsr3
file << mur
