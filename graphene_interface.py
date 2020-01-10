'''
Author: Zhiwei Fang, Lehigh University
Email: zhiweifang1987@gmail.com
Copyright reserved
Simulation of graphene SPP phenomenon by using interface method,
which means the graphene is assumed to be no height.
'''

from dolfin import *
import numpy as np
import sympy as syp
from math import pi, log, sqrt, exp, log1p
Nx = Ny = 200
nPML = 12
ax0 = -30*1e-6
bx0 = 30*1e-6
ay0 = -10*1e-6
by0 = 10*1e-6
ax = ax0 - nPML*(bx0-ax0)/Nx
bx = bx0 + nPML*(bx0-ax0)/Nx
ay = ay0 - nPML*(by0-ay0)/Ny
by = by0 + nPML*(by0-ay0)/Ny
cc = 2.99792458e8
mu = 4.0*pi*1.0e-7
eps = 1.0/(cc*cc*mu)
Nt = 2000
dt = (by0-ay0)/Ny/4.0/cc
T = dt*Nt
sw_f0 = 1e13
q = 1.60217662*1e-19
h_bar = 1.054571817*1e-34
kB = 1.38064852*1e-23
Tg = 300
muc = 2*1.5*1.60218e-19
tau = 1.2e-12
mu = 4*pi*1e-7
ep = 8.854187817*1e-12
sg0 = q**2*muc*tau/(pi*h_bar**2)
mesh = RectangleMesh(Point(ax, ay), Point(bx, by), Nx+2*nPML, Ny+2*nPML)
If_Refine = False
if If_Refine:
	tol_refine = 1e-1*(by0 - ay0)
	class Omega_refine_cls(SubDomain):
		def inside(self, x, on_boundary):
			return (between(x[0], (ax0, bx0)) and between(x[1], (-tol_refine, tol_refine)) )
	Omega_refine = Omega_refine_cls()
	refine_markers = MeshFunction("bool", mesh, mesh.topology().dim())
	refine_markers.set_all(False)
	Omega_refine.mark(refine_markers, True)
	mesh = refine(mesh, refine_markers)
tol = 5e-3*(by0 - ay0)
class Omega_cls_3(SubDomain):
    def inside(self, x, on_boundary):
      return (between(x[0], (ax0, bx0)) and between(x[1], (-tol,tol)))

class Omega_cls_2(SubDomain):
    def __init__(self, ddict):
        self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
        self.ymin, self.ymax = ddict['ymin'], ddict['ymax']
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        bol1 = (between(x[0], (self.xmin, self.xmax)) and between(x[1], (self.ymin, self.ymax)))
        return bol1

cdcoo = dict(xmin=ax0, xmax=bx0, ymin=ay0, ymax=by0)
Omega_2 = Omega_cls_2(cdcoo)
Omega_3 = Omega_cls_3()


subdom = MeshFunction("size_t", mesh, mesh.topology().dim())
subdom.set_all(1)
Omega_2.mark(subdom, 2)
Omega_3.mark(subdom, 3)

dx = Measure('dx', domain=mesh, subdomain_data=subdom)

V = FiniteElement("N2curl", mesh.ufl_cell(), 1)
U = FiniteElement("DG", mesh.ufl_cell(), 0)
Vs = FunctionSpace(mesh, V)
Us = FunctionSpace(mesh, U)

def boundary(x, on_boundary):
    return on_boundary
bc_bnd = DirichletBC(Vs, Constant((0.0, 0.0)), boundary)
c_scr = 2*pi*sw_f0
Js_x_str = '0.0'
Js_y_str = '0.0'
Js = Expression((Js_x_str, Js_y_str), degree=1, h=(by0-ay0)/Ny, tol1 = (1e-3*(by0 - ay0))**2, c_scr=c_scr, t=0.0)
Ks_str = '(pow(x[0]+27e-6, 2) + pow(x[1]-1e-6, 2)<tol1) ? sin(c_scr*t)/h : ((pow(x[0]+27e-6, 2) + pow(x[1]+1e-6, 2)<tol1) ? -sin(c_scr*t)/h : 0.0)'
Ks = Expression(Ks_str, degree=1, h=(by0-ay0)/Ny, tol1 = (1e-4*(by0 - ay0))**2, c_scr=c_scr, t=0.0)
phy_mark = []
dm = Us.dofmap()
tol_s = 1e-3
for c in cells(mesh):
    x = c.midpoint().x()
    y = c.midpoint().y()
    if subdom[c] > 1:
	    phy_mark.extend(dm.cell_dofs(c.index()))
phy_mark = list(set(phy_mark))
E0 = Constant((0.0, 0.0))
H0 = Constant(0.0)
J0 = Constant((0.0, 0.0))
K0 = Constant(0.0)
cppcode_Coef_dim1 = """
class Coef_dim1 : public Expression
{
public:

  void eval(Array<double>& values,
            const Array<double>& x,
            const ufc::cell& cell) const
  {
    if ((*subdom)[cell.index] <2)
      values[0] = Coef_dim1_fcn1;
    else
      values[0] = Coef_dim1_fcn23;
  }

  std::shared_ptr<MeshFunction<std::size_t>> subdom;
};
"""
cppcode_Coef_dim22 = """
class Coef_dim22 : public Expression
{
public:

  // Create expression with 3 components
  Coef_dim22() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    if ((*subdom)[cell.index] <2)
	{
      values[0] = Coef_dim22_fcn1_e1;
	  values[1] = 0.0;
	  values[2] = Coef_dim22_fcn1_e2;
	}
    else
	{
      values[0] = Coef_dim22_fcn23_e1;
	  values[1] = 0.0;
	  values[2] = Coef_dim22_fcn23_e2;
	}
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<std::size_t>> subdom;
};
"""
err = 1e-7
polyn = 4.0
ddx = nPML*(bx0-ax0)/Nx
ddy = nPML*(by0-ay0)/Ny
smx = -log(err)*(polyn+1)*cc / (2.0*ddx)
smy = -log(err)*(polyn+1)*cc / (2.0*ddy)
sx_str = "(x[0]>=bx0 ? smx*pow((x[0]-bx0)/ddx, polyn) : (x[0]<=ax0 ? smx*pow((ax0-x[0])/ddx, polyn) : 0.0))"
sy_str = "(x[1]>=by0 ? smy*pow((x[1]-by0)/ddy, polyn) : (x[1]<=ay0 ? smy*pow((ay0-x[1])/ddy, polyn) : 0.0))"
def StrPro(str1, dict0):
    str0 = []
    for var1 in str1:
        str0.append(StrDictPro(var1.replace('sx_str',sx_str).replace('sy_str',sy_str), dict0))
    return str0
def StrDictPro(str1, dict0):
    for k in dict0.keys():
	    str1 = str1.replace(str(k), str(dict0[k]))
    return str1
para_dict = {'eps':eps, 'mu':mu, 'smx':smx, 'smy':smy, 'ddx':ddx, 'ddy':ddy, 'ax0':ax0, 'bx0':bx0, 'ay0':ay0, 'by0':by0, 'polyn':polyn, 'dt':dt, 'tau':tau, 'sg0':sg0}
E1f1e1_str = "eps/dt+0.5*eps*(sy_str-sx_str)"
E1f1e2_str = "eps/dt+0.5*eps*(sx_str-sy_str)"
[E1f1e1_str, E1f1e2_str] = StrPro([E1f1e1_str, E1f1e2_str], para_dict)
E1_f23 = eps/dt
cE1_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : E1f1e1_str, 'Coef_dim22_fcn23_e1' : E1_f23, 'Coef_dim22_fcn1_e2' : E1f1e2_str, 'Coef_dim22_fcn23_e2' : E1_f23})
cE1_raw = Expression(cE1_code, subdom=subdom, degree = int(polyn))
cE1 = as_matrix(((cE1_raw[0], cE1_raw[1]), (cE1_raw[1], cE1_raw[2])))

E2f1e1_str = "eps/dt-0.5*eps*(sy_str-sx_str)"
E2f1e2_str = "eps/dt-0.5*eps*(sx_str-sy_str)"
[E2f1e1_str, E2f1e2_str] = StrPro([E2f1e1_str, E2f1e2_str], para_dict)
E2_f23 = eps/dt
cE2_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : E2f1e1_str, 'Coef_dim22_fcn23_e1' : E2_f23, 'Coef_dim22_fcn1_e2' : E2f1e2_str, 'Coef_dim22_fcn23_e2' : E2_f23})
cE2_raw = Expression(cE2_code, subdom=subdom, degree = int(polyn))
cE2 = as_matrix(((cE2_raw[0], cE2_raw[1]), (cE2_raw[1], cE2_raw[2])))

D2_f1 = 1.0
D2_f2 = 0.0
cD2_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : D2_f1, 'Coef_dim1_fcn23' : D2_f2})
cD2 = Expression(cD2_code, subdom=subdom, element = U)

J1f1e1_str = "(2-dt*sx_str)/(2+dt*sx_str)"
J1f1e2_str = "(2-dt*sy_str)/(2+dt*sy_str)"
[J1f1e1_str, J1f1e2_str] = StrPro([J1f1e1_str, J1f1e2_str], para_dict)
J1_f23 = (2*tau-dt)/(2*tau+dt)
cJ1_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : J1f1e1_str, 'Coef_dim22_fcn23_e1' : J1_f23, 'Coef_dim22_fcn1_e2' : J1f1e2_str, 'Coef_dim22_fcn23_e2' : J1_f23})
cJ1_raw = Expression(cJ1_code, subdom=subdom, degree = int(polyn))
cJ1 = as_matrix(((cJ1_raw[0], cJ1_raw[1]), (cJ1_raw[1], cJ1_raw[2])))

J2f1e1_str = "2*dt*eps*sx_str*(sx_str-sy_str)/(2+dt*sx_str)"
J2f1e2_str = "2*dt*eps*sy_str*(sy_str-sx_str)/(2+dt*sy_str)"
[J2f1e1_str, J2f1e2_str] = StrPro([J2f1e1_str, J2f1e2_str], para_dict)
J2_f23 = 2*dt*sg0/(2*tau+dt)
cJ2_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : J2f1e1_str, 'Coef_dim22_fcn23_e1' : J2_f23, 'Coef_dim22_fcn1_e2' : J2f1e2_str, 'Coef_dim22_fcn23_e2' : J2_f23})
cJ2_raw = Expression(cJ2_code, subdom=subdom, degree = int(polyn))
cJ2 = as_matrix(((cJ2_raw[0], cJ2_raw[1]), (cJ2_raw[1], cJ2_raw[2])))

H1f1_str = "(2-dt*(sx_str+sy_str))/(2+dt*(sx_str+sy_str))"
[H1f1_str] = StrPro([H1f1_str], para_dict)
H1f2 = 1.0
cH1_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : H1f1_str, 'Coef_dim1_fcn23' : H1f2})
cH1 = Expression(cH1_code, subdom=subdom, element = U)

H2f1_str = "2*dt/(2*mu+dt*mu*(sx_str+sy_str))"
[H2f1_str] = StrPro([H2f1_str], para_dict)
H2f2 = dt/mu
cH2_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : H2f1_str, 'Coef_dim1_fcn23' : H2f2})
cH2 = Expression(cH2_code, subdom=subdom, element = U)

H3f1_str = "2*dt/(2*mu+dt*mu*(sx_str+sy_str))"
[H3f1_str] = StrPro([H3f1_str], para_dict)
H3f2 = 0.0
cH3_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : H3f1_str, 'Coef_dim1_fcn23' : H3f2})
cH3 = Expression(cH3_code, subdom=subdom, element = U)

K1f1_str = "dt*mu*sx_str*sy_str"
[K1f1_str] = StrPro([K1f1_str], para_dict)
K1f2 = 0.0
cK1_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : K1f1_str, 'Coef_dim1_fcn23' : K1f2})
cK1 = Expression(cK1_code, subdom=subdom, element = U)

E0 = interpolate(E0, Vs)
H0 = interpolate(H0, Us)
J0 = interpolate(J0, Vs)
K0 = interpolate(K0, Us)
E = interpolate(E0, Vs)
H = interpolate(H0, Us)
J = interpolate(J0, Vs)
K = interpolate(K0, Us)
E = TrialFunction(Vs)
psi= TestFunction(Vs)
F = -inner(cE1*E,psi)*dx + inner(cE2*E0,psi)*dx + inner(H0, curl(psi))*dx + inner(J0*0.5/tol, psi)*dx(3) \
- inner(cD2*J0, psi)*dx - inner(Js, psi)*dx
a, L = lhs(F), rhs(F)
A = assemble(a)
b = None
E = Function(Vs)
H_st = interpolate(Constant(0.0), Us)

t = 0.0
vtkfile = File('graphene_rect/solution.pvd')

from time import *
import matplotlib.pyplot as plt
for n in range(Nt):
    print 'time step is: %d (total steps: %d)' % (n, Nt)
    t += dt
    s0 = clock()
    if (n>0):
        H0.assign(H)
        E0.assign(E)
        K0.assign(K)
        J0.assign(J)
	Js.t = t + 0.5*dt
	Ks.t = t + dt
    b = assemble(L, tensor=b)
    bc_bnd.apply(A, b)
    solve(A, E.vector(), b)

    J.assign(project(cJ1*J0 + cJ2*E, Vs))

    K.assign(project(K0 + cK1*H0, Us))

    H.assign(project(cH1*H0 - cH2*curl(E) - cH3*K - cH2*Ks, Us))

    H_st.vector()[phy_mark] = H.vector()[phy_mark]
    e0 = clock()
    print e0 - s0
    if (n%50==0):
      vtkfile << (H_st, t)
