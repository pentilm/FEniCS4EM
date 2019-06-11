# Edge element with CN scheme for backward wave propagation.
# For the physical model, please see:
# Huang, Y., Li, J. and Yang, W., 2013. Modeling backward wave propagation in metamaterials by the finite element time-domain method. SIAM Journal on Scientific Computing, 35(1), pp.B248-B274.
# Author: Zhiwei Fang
# Copyright reserved

from dolfin import *
import numpy as np
import sympy as syp
#from mshr import *
from math import pi, log, sqrt
# Computational Parameters:
T = 1e-9
Nt = 1000
Nx = Ny = 100
dt = T/Nt
nPML = 8
ax0 = ay0 = 0.0
bx0 = 0.07
by0 = 0.064
ax = ax0 - nPML*(bx0-ax0)/Nx
bx = bx0 + nPML*(bx0-ax0)/Nx
ay = ay0 - nPML*(by0-ay0)/Ny
by = by0 + nPML*(by0-ay0)/Ny
# Physical Parameters:
cc = 2.99792458e8
mu = 4.0*pi*1.0e-7
eps = 1.0/(cc*cc*mu)
sw_f0 = 3e10
ge = gm = 1e8
we = wm = 2 * pi*sqrt(5)*sw_f0
# we = wm = 2 * pi*sqrt(2)*sw_f0
# Define the domain and mesh
#domain = Rectangle(Point(ax, ay), Point(bx, by))
#mesh = generate_mesh(domain, Nx*Ny)
mesh = RectangleMesh(Point(ax, ay), Point(bx, by), Nx+2*nPML, Ny+2*nPML)
If_Refine_Meta = True	# The marker if refine metamaterial
# Subdomain of metamaterial
tol = 1e-5
class Omega_cls_3(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (0.024-tol, 0.044+tol)) and between(x[1], (0.002-tol, 0.062+tol)))
        #return (x[0]>=0.024-DOLFIN_EPS) and (x[0]<=0.044+DOLFIN_EPS) and (x[1]>=0.002-DOLFIN_EPS) and (x[1]<=0.062+DOLFIN_EPS)

class Omega_cls_2(SubDomain):
    def __init__(self, ddict):
		self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
		self.ymin, self.ymax = ddict['ymin'], ddict['ymax']
		SubDomain.__init__(self)
    def inside(self, x, on_boundary):
		# in physical domain
		bol1 = (between(x[0], (self.xmin, self.xmax)) and between(x[1], (self.ymin, self.ymax)))
		# in metamaterial
		bol2 = ((between(x[0], (0.024+tol, 0.044-tol)) and between(x[1], (0.002+tol, 0.062-tol))))
		return (bol1 and (not bol2))	# in physical domain but not in metamaterial

cdcoo = dict(xmin=ax0, xmax=bx0, ymin=ay0, ymax=by0)
# Omega_1 = Omega_cls_1(cdcoo)
Omega_2 = Omega_cls_2(cdcoo)
Omega_3 = Omega_cls_3()
# Refine the metamaterial if is required
if If_Refine_Meta:
	refine_markers = MeshFunction("bool", mesh, mesh.topology().dim())
	refine_markers.set_all(False)
	Omega_3.mark(refine_markers, True)
	mesh = refine(mesh, refine_markers)

subdom = MeshFunction("size_t", mesh, mesh.topology().dim())
subdom.set_all(1)
Omega_2.mark(subdom, 2)
Omega_3.mark(subdom, 3)

# Define the function spaces
V = FiniteElement("N2curl", mesh.ufl_cell(), 1)	# space for (Hx, Hy)
U = FiniteElement("DG", mesh.ufl_cell(), 0)			# space for E
Vs = FunctionSpace(mesh, V)
Us = FunctionSpace(mesh, U)
N1L2 = V*U
W = FunctionSpace(mesh, N1L2)

def boundary(x, on_boundary):
    return on_boundary
bc_bnd = DirichletBC(W, Constant((0.0, 0.0, 0.0)), boundary)
# define the class of source function
class Source_cls(Expression):
    def __init__(self, f0, dx, t, **kwargs):
        self.f0 = f0
        self.dx = dx
        self.t = t

    def eval(self, values, x):
        k = 100.0
        m = 2.0
        w0 = 2.0*pi*self.f0
        Tp = 1/self.f0
        x1 = self.t/m/Tp
        x2 = (self.t - (m + k)*Tp)/m/Tp
        valy = exp(-(x[1] - 0.03)*(x[1] - 0.03)/2500.0/self.dx/self.dx)
        if self.t <= m*Tp and self.t>=0.0:
            values[0] = valy*(10.0*pow(x1, 3)-15.0*pow(x1, 4)+6.0*pow(x1, 5))*sin(w0*self.t)
        elif self.t <= (m+k)*Tp and self.t>m*Tp:
            values[0] = valy*sin(w0*self.t)
        elif self.t <= (2.0*m+k)*Tp and self.t>(m+k)*Tp:
            values[0] = valy*(1.0-(10.0*pow(x2, 3)-15.0*pow(x2, 4)+6.0*pow(x2, 5)))*sin(w0*self.t)
        else:
            values[0] = 0.0
Source = Source_cls(sw_f0, (bx0-ax0)/Nx, 0.0, element = U)	# initialize the source function
# set the DoF marksers for source and physical domain
scr_mark = []	# marker for source of E
phy_mark = []   # marker for physical domain
dm = Us.dofmap()
tol_s = 1e-3
for c in cells(mesh):
    x = c.midpoint().x()
    y = c.midpoint().y()
    if near(x, 0.004, tol_s) and between(y, (0.025, 0.035)):
        scr_mark.extend(dm.cell_dofs(c.index()))
    if subdom[c] > 1:
	    phy_mark.extend(dm.cell_dofs(c.index()))
scr_mark = list(set(scr_mark))
phy_mark = list(set(phy_mark))
# define the functions
H0 = Constant((0.0, 0.0))
E0 = Constant(0.0)
K0 = Constant((0.0, 0.0))
J0 = Constant(0.0)
rhs_f1 = Constant(0.0)
rhs_f2 = Constant((0.0, 0.0))
rhs_f3 = Constant(0.0)
rhs_f4 = Constant((0.0, 0.0))
# prototype of coefficient functions
# cpp code for piecewise defined scalar function
# Coef_dim1_fcn1, Coef_dim1_fcn2 and Coef_dim1_fcn3 must be replaced when use
cppcode_Coef_dim1 = """
class Coef_dim1 : public Expression
{
public:

  void eval(Array<double>& values,
            const Array<double>& x,
            const ufc::cell& cell) const
  {
    if ((*subdom)[cell.index] == 1)
      values[0] = Coef_dim1_fcn1;
	else if  ((*subdom)[cell.index] == 2)
	  values[0] = Coef_dim1_fcn2;
    else
      values[0] = Coef_dim1_fcn3;
  }

  std::shared_ptr<MeshFunction<std::size_t>> subdom;
};
"""
# cpp code for piecewise defined 2x2 diagonal matrix function
# Coef_dim22_fcn1_e1, Coef_dim22_fcn1_e2, Coef_dim22_fcn2_e1, Coef_dim22_fcn2_e2
# Coef_dim22_fcn3_e1 and Coef_dim22_fcn3_e2 must be replaced when use
cppcode_Coef_dim22 = """
class Coef_dim22 : public Expression
{
public:

  // Create expression with 3 components
  Coef_dim22() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    if ((*subdom)[cell.index] == 1)
	{
      values[0] = Coef_dim22_fcn1_e1;
	  values[1] = 0.0;
	  values[2] = Coef_dim22_fcn1_e2;
	}
	else if ((*subdom)[cell.index] == 2)
	{
	  values[0] = Coef_dim22_fcn2_e1;
	  values[1] = 0.0;
	  values[2] = Coef_dim22_fcn2_e2;
	}
    else
	{
      values[0] = Coef_dim22_fcn3_e1;
	  values[1] = 0.0;
	  values[2] = Coef_dim22_fcn3_e2;
	}
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<std::size_t>> subdom;
};
"""
# parameters of PML
err = 1e-7
polyn = 4.0
ddx = nPML*(bx0-ax0)/Nx
ddy = nPML*(by0-ay0)/Ny
smx = -log(err)*(polyn+1)*cc / (2.0*ddx)
smy = -log(err)*(polyn+1)*cc / (2.0*ddy)
sx_str = "(x[0]>=bx0 ? smx*pow((x[0]-bx0)/ddx, polyn) : (x[0]<=ax0 ? smx*pow((ax0-x[0])/ddx, polyn) : 0.0))"
sy_str = "(x[1]>=by0 ? smy*pow((x[1]-by0)/ddy, polyn) : (x[1]<=ay0 ? smy*pow((ay0-x[1])/ddy, polyn) : 0.0))"
# batch processing function for string of function
def StrPro(str1, dict0):
# str1 is the string going to handle, str2 is the string for [sx_str,sy_str]
    str0 = []
    for var1 in str1:
        str0.append(StrDictPro(var1.replace('sx_str',sx_str).replace('sy_str',sy_str), dict0))
    return str0
# batch processing function for string of function by using a dictionary
def StrDictPro(str1, dict0):
# str1 is the string going to handle, dict is the dictionary to repalce the string
    for k in dict0.keys():
	    str1 = str1.replace(str(k), str(dict0[k]))
    return str1
para_dict = {'eps':eps, 'mu':mu, 'smx':smx, 'smy':smy, 'ddx':ddx, 'ddy':ddy, 'ax0':ax0, 'bx0':bx0, 'ay0':ay0, 'by0':by0, 'polyn':polyn, 'dt':dt}
# coefficient functions
E1f1_str = "eps*(2.0/dt+(sx_str+sy_str)+0.5*dt*sx_str*sy_str)"
[E1f1_str] = StrPro([E1f1_str], para_dict)
E1f2 = 2.0*eps/dt
E1f3 = eps*(2.0/dt+we*we/(2.0/dt+ge))
cE1_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : E1f1_str, 'Coef_dim1_fcn2' : E1f2, 'Coef_dim1_fcn3' : E1f3})
cE1 = Expression(cE1_code, subdom=subdom, degree = int(polyn))

E2f1_str = "eps*(2.0/dt-(sx_str+sy_str)-0.5*dt*sx_str*sy_str)"
[E2f1_str] = StrPro([E2f1_str], para_dict)
E2f2 = 2.0*eps/dt
E2f3 = eps*(2.0/dt-we*we/(2.0/dt+ge))
cE2_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : E2f1_str, 'Coef_dim1_fcn2' : E2f2, 'Coef_dim1_fcn3' : E2f3})
cE2 = Expression(cE2_code, subdom=subdom, degree = int(polyn))

cEJ_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : 2.0, 'Coef_dim1_fcn2' : 2.0, 'Coef_dim1_fcn3' : 4.0/dt/(2.0/dt+ge)})
cEJ = Expression(cEJ_code, subdom=subdom, element = U)

cJ1_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : 1.0, 'Coef_dim1_fcn2' : 1.0, 'Coef_dim1_fcn3' : (2.0/dt-ge)/(2.0/dt+ge)})
cJ1 = Expression(cJ1_code, subdom=subdom, element = U)

J2f1_str = "0.5*eps*dt*sx_str*sy_str"
[J2f1_str] = StrPro([J2f1_str], para_dict)
cJ2_code = StrDictPro(cppcode_Coef_dim1, {'Coef_dim1_fcn1' : J2f1_str, 'Coef_dim1_fcn2' : 0.0, 'Coef_dim1_fcn3' : eps*we**2/(2.0/dt+ge)})
cJ2 = Expression(cJ2_code, subdom=subdom, degree = int(polyn))

H1f1e1_str = "mu*(2.0/dt+(sy_str-sx_str)+sx_str*(sx_str-sy_str)/(2.0/dt+sx_str))"
H1f1e2_str = "mu*(2.0/dt+(sx_str-sy_str)+sy_str*(sy_str-sx_str)/(2.0/dt+sy_str))"
[H1f1e1_str, H1f1e2_str] = StrPro([H1f1e1_str, H1f1e2_str], para_dict)
H1f2 = 2.0*mu/dt
H1f3 = mu*(2.0/dt+wm*wm/(2.0/dt+gm))
cH1_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : H1f1e1_str, 'Coef_dim22_fcn1_e2' : H1f1e2_str, 'Coef_dim22_fcn2_e1' : H1f2, 'Coef_dim22_fcn2_e2' : H1f2, 'Coef_dim22_fcn3_e1' : H1f3, 'Coef_dim22_fcn3_e2' : H1f3})
cH1_raw = Expression(cH1_code, subdom=subdom, degree = int(polyn))
cH1 = as_matrix(((cH1_raw[0], cH1_raw[1]), (cH1_raw[1], cH1_raw[2])))

H2f1e1_str = "mu*(2.0/dt-(sy_str-sx_str)-sx_str*(sx_str-sy_str)/(2.0/dt+sx_str))"
H2f1e2_str = "mu*(2.0/dt-(sx_str-sy_str)-sy_str*(sy_str-sx_str)/(2.0/dt+sy_str))"
[H2f1e1_str, H2f1e2_str] = StrPro([H2f1e1_str, H2f1e2_str], para_dict)
H2f2 = 2.0*mu/dt
H2f3 = mu*(2.0/dt-wm*wm/(2.0/dt+gm))
cH2_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : H2f1e1_str, 'Coef_dim22_fcn1_e2' : H2f1e2_str, 'Coef_dim22_fcn2_e1' : H2f2, 'Coef_dim22_fcn2_e2' : H2f2, 'Coef_dim22_fcn3_e1' : H2f3, 'Coef_dim22_fcn3_e2' : H2f3})
cH2_raw = Expression(cH2_code, subdom=subdom, degree = int(polyn))
cH2 = as_matrix(((cH2_raw[0], cH2_raw[1]), (cH2_raw[1], cH2_raw[2])))

K1f1e1_str = "(2.0/dt-sx_str)/(2.0/dt+sx_str)"
K1f1e2_str = "(2.0/dt-sy_str)/(2.0/dt+sy_str)"
[K1f1e1_str, K1f1e2_str] = StrPro([K1f1e1_str, K1f1e2_str], para_dict)
K1f2 = 1.0
K1f3 = (2.0/dt-gm)/(2.0/dt+gm)
cK1_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : K1f1e1_str, 'Coef_dim22_fcn1_e2' : K1f1e2_str, 'Coef_dim22_fcn2_e1' : K1f2, 'Coef_dim22_fcn2_e2' : K1f2, 'Coef_dim22_fcn3_e1' : K1f3, 'Coef_dim22_fcn3_e2' : K1f3})
cK1_raw = Expression(cK1_code, subdom=subdom, degree = int(polyn))
cK1 = as_matrix(((cK1_raw[0], cK1_raw[1]), (cK1_raw[1], cK1_raw[2])))

K2f1e1_str = "mu*sx_str*(sx_str-sy_str)/(2.0/dt+sx_str)"
K2f1e2_str = "mu*sy_str*(sy_str-sx_str)/(2.0/dt+sy_str)"
[K2f1e1_str, K2f1e2_str] = StrPro([K2f1e1_str, K2f1e2_str], para_dict)
K2f2 = 0.0
K2f3 = mu*wm*wm/(2.0/dt+gm)
cK2_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : K2f1e1_str, 'Coef_dim22_fcn1_e2' : K2f1e2_str, 'Coef_dim22_fcn2_e1' : K2f2, 'Coef_dim22_fcn2_e2' : K2f2, 'Coef_dim22_fcn3_e1' : K2f3, 'Coef_dim22_fcn3_e2' : K2f3})
cK2_raw = Expression(cK2_code, subdom=subdom, degree = int(polyn))
cK2 = as_matrix(((cK2_raw[0], cK2_raw[1]), (cK2_raw[1], cK2_raw[2])))

HKf1e1_str = "4.0/dt/(2.0/dt+sx_str)"
HKf1e2_str = "4.0/dt/(2.0/dt+sy_str)"
[HKf1e1_str, HKf1e2_str] = StrPro([HKf1e1_str, HKf1e2_str], para_dict)
HKf2 = 2.0
HKf3 = 4.0/dt/(2.0/dt+gm)
cHK_code = StrDictPro(cppcode_Coef_dim22, {'Coef_dim22_fcn1_e1' : HKf1e1_str, 'Coef_dim22_fcn1_e2' : HKf1e2_str, 'Coef_dim22_fcn2_e1' : HKf2, 'Coef_dim22_fcn2_e2' : HKf2, 'Coef_dim22_fcn3_e1' : HKf3, 'Coef_dim22_fcn3_e2' : HKf3})
cHK_raw = Expression(cHK_code, subdom=subdom, degree = int(polyn))
cHK = as_matrix(((cHK_raw[0], cHK_raw[1]), (cHK_raw[1], cHK_raw[2])))
# setup for FEM
H0 = interpolate(H0, Vs)
E0 = interpolate(E0, Us)
J0 = interpolate(J0, Us)
K0 = interpolate(K0, Vs)
(H, E) = TrialFunctions(W)
(psi, phi) = TestFunctions(W)
# form weak formulation of H and E
F = inner(cH1*H, psi)*dx + E*curl(psi)*dx \
- inner(cH2*H0, psi)*dx + E0*curl(psi)*dx + inner(cHK*K0, psi)*dx \
+ cE1*E*phi*dx - curl(H)*phi*dx \
- cE2*E0*phi*dx - curl(H0)*phi*dx + cEJ*J0*phi*dx
a, L = lhs(F), rhs(F)
A = assemble(a)
b = None
u = Function(W)
E_st = interpolate(Constant(0.0), Us)	# E field on physical domain used to store

t = 0.0
vtkfile = File('meta/solution.pvd')
from time import *
for n in range(Nt):
    print 'time step is: %d (total steps: %d)' % (n, Nt)
    t += dt
    s0 = clock()
	# update the time of RHS if there is any
    b = assemble(L, tensor=b)
    bc_bnd.apply(A, b)
    solve(A, u.vector(), b)
    H, E = u.split(deepcopy = True)
	# Update the source
    Source.t = t
    src_fcn = interpolate(Source, Us)
    E.vector()[scr_mark] = src_fcn.vector()[scr_mark]
	# Update the J and K
    J0.assign(project(cJ1*J0 + cJ2*(E+E0), Us))
    K0.assign(project(cK1*K0 + cK2*(H+H0), Vs))
	# update and write solution
    H0.assign(H)
    E0.assign(E)
	# extract DoFs in subdomains
    E_st.vector()[phy_mark] = np.abs(E.vector()[phy_mark])
    e0 = clock()
    print e0 - s0
    vtkfile << (E_st, t)

# # # import matplotlib.pyplot as plt
# # # plt.figure()
# # # plot(u0)
# # # plt.show()

# import MyPlot as mp
# mp.MyTriSurf(mesh, u0)
