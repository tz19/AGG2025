from fenics import *
from ufl import *
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import numpy as np
import math
from dolfin import *

if MPI.rank(MPI.comm_world) > 0:
    set_log_level(ERROR)

parameters["form_compiler"]["quadrature_degree"] = 2

Lx = 400
Nx = 400
W = Lx/Nx
C11 = 246.5E3
C12 = 147.3E3
C44 = 124.7E3
dLx = 1.0E-10
num_phi = 16 # Divide 45 degrees into # of parts and rotate them along z-axis
num_grain = 1000 # Number of grains generate at the beginning
mu    = 1.0 # free energy constant
kappa = 1.0 # energy gradient coefficient
L  = 1.0

# Define elastic tensor: Refer to Micromechanics-Materials textbook P25
dijkl = []
for i in range(0,3):
    dijkl += [[],]
    for j in range(0,3):
        dijkl[i] += [[],]
        for k in range(0,3):
            dijkl[i][j] += [[],]
            for l in range(0,3):
                    dijkl[i][j][k] += [0.0,]
dijkl[0][0][0][0] = 1.0
dijkl[1][1][1][1] = 1.0
dijkl[2][2][2][2] = 1.0
dijkl = as_tensor(dijkl)

i, j, k, l = indices(4)
I = Identity(3)
C0 = as_tensor(C12*I[i,j]*I[k,l] + C44*(I[i,k]*I[j,l]+I[i,l]*I[j,k]) , (i,j,k,l))
C0 = C0 + (C11-C12-2*C44)*dijkl

# Rotate the elastic tensor
C_list = [] 
gamma_list = [] 

for ii in range(num_phi):
    i, j, k, l, p, q, r, s = indices(8)
    rot = Rotation.from_euler('zxz', [45*ii/(num_phi-1), 0, 0], degrees=True)
    Q = as_tensor(rot.as_matrix())
    Ctmp = as_tensor(C0[p,q,r,s]*Q[p,i]*Q[q,j]*Q[r,k]*Q[s,l], (i,j,k,l))
    C_list.append(Ctmp)
    gamma_list.append(Q)

C_list = as_tensor(C_list) # stiffness tensors after roation
gamma_list = as_tensor(gamma_list) #rotation matrices

# Define mesh

mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Lx, W), Nx, Nx, 1)

S = FunctionSpace(mesh, 'CG', 1)
V  = VectorFunctionSpace(mesh, 'CG', 1)
T  = TensorFunctionSpace(mesh, 'CG', 1)
PF = VectorFunctionSpace(mesh, 'CG', 1, dim=num_phi)

# Define Voronoi
points = []
x = np.sqrt(1/num_grain)
for i in range(int(1/x)+2):
    for j in range(int(1/x)+2):
        px = np.random.uniform(i*x, (i+1)*x)
        py = np.random.uniform(j*x, (j+1)*x)
        points.append([px, py])       
points = np.array(points)
np.random.shuffle(points)

# points = np.random.rand(num_grain,2) 
tree = KDTree(points)

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval(self, values, x):
        distance, grain_index = tree.query([x[0]/Lx, x[1]/Lx])
        for i in range(num_phi):
            values[i] = 0.0
        values[grain_index % num_phi] = 1.0*math.exp(-1.0*(distance/0.1)**2) # Normal distribution for better convergence
    def value_shape(self):
        return (num_phi,)

v     = TestFunction(V)
u     = Function(V)
du    = TrialFunction(V)
Gamma = Function(T, name='Gamma')

phi   = Function(PF, name='order parameters') # current phase field variable, vector
phi_n = Function(PF) # history of phase field variable
q     = TestFunction(PF)
dq    = TrialFunction(PF)

phi_init = InitialConditions(degree=1)
phi.interpolate(phi_init)
phi_n.assign(phi)

tol = 1E-10

def BC_x0(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)
def BC_x1(x, on_boundary):
    return on_boundary and near(x[0], Lx, tol)
def BC_y0(x, on_boundary):
    return on_boundary and near(x[1], 0, tol)
def BC_z0(x, on_boundary):
    return on_boundary and near(x[2], 0, tol)
def BC_origin(x, on_boundary):
    return on_boundary and near(x[0], 0, tol) and near(x[1], 0, tol) and near(x[2], 0, tol)

bc1 = DirichletBC(V.sub(0), 0,    BC_x0)
bc2 = DirichletBC(V.sub(0), dLx,  BC_x1)
bc3 = DirichletBC(V.sub(1), 0,    BC_y0)
bc4 = DirichletBC(V.sub(2), 0,    BC_z0)
bc5 = DirichletBC(V.sub(1), 0,    BC_origin)
bc6 = DirichletBC(V.sub(2), 0,    BC_origin)
# bc = [bc1, bc2, bc3, bc4] # much more stable
bc = [bc1, bc2, bc5, bc6]

# Small Strain tensor
def eps(u):
    return sym(grad(u))

# Stress tensor
def sigma(u, phi):
    i, j, k, l= indices(4)
    e = eps(u)

    h_phi = h(phi)
    C = dot(h_phi, C_list) # The elastic stiffness tensor of the whole sample.
    gamma = dot(h_phi, gamma_list)

    return C, gamma, as_tensor(C[i,j,k,l]*e[k,l], (i,j))

# weight on each elastic stiffness tensor corresponding to rotation angle --  colume vector 
def h(phi):
    h_phi = []
    h_tot = 0
    for i in range(num_phi):
        h_phi.append(0.5*(1+sin(pi*(phi[i]-0.5))))
        h_tot = h_tot + 0.5*(1+sin(pi*(phi[i]-0.5)))

    return as_tensor(h_phi)/h_tot

# Free energy density (part of)
def free_energy(phi):
    f = 0
    for i in range(num_phi):
        f = f + mu*(pow(phi[i],4)/4-pow(phi[i],2)/2)
        for j in range(num_phi):
            if j > i:
                f = f + 1.5*mu*pow(phi[i],2)*pow(phi[j],2)
    return f


# Class for interfacing with the Newton solver
class PhaseField(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

solver = NewtonSolver()
solver.parameters["linear_solver"] = "cg" #"lu" "mumps"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-8
solver.parameters["maximum_iterations"] = 1000
solver.parameters["error_on_nonconvergence"] = False

file_phi = XDMFFile('output/phi.xdmf')
file_phi.write(mesh)
file_phi.parameters["flush_output"] = True

file_W = XDMFFile('output/W_elas.xdmf')
file_W.write(mesh)
file_W.parameters["flush_output"] = True

file_sigma = XDMFFile('output/stress.xdmf')
file_sigma.write(mesh)
file_sigma.parameters["flush_output"] = True

file_uij = XDMFFile('output/uij.xdmf')
file_uij.write(mesh)
file_uij.parameters["flush_output"] = True


dt = 0.1
t = 0
i, j, k, l, m = indices(5)

inc_max = 502
for inc in range(inc_max):
    C, gamma, sig = sigma(u,phi)
    dCdphi = diff(C, phi)
    dFdphi = diff(free_energy(phi), phi)


    F = inner(sig, eps(v))*dx
    if t > 5:
        dLx = 1.0
        bc2 = DirichletBC(V.sub(0), dLx,  BC_x1)
        # bc = [bc1, bc2, bc3, bc4] # much more stable
        bc = [bc1, bc2, bc5, bc6]

    J = derivative(F, u, du)
    problem_L = NonlinearVariationalProblem(F, u, bc, J)
    solver_L = NonlinearVariationalSolver(problem_L)
    prm = solver_L.parameters 
    prm["newton_solver"]["linear_solver"] = "cg"
    prm["newton_solver"]["absolute_tolerance"] = 1E-9 
    prm["newton_solver"]["relative_tolerance"] = 1E-8 
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    prm["newton_solver"]["error_on_nonconvergence"] = False
    solver_L.solve()

    # solve(F == 0, u, bc) # Solve for linear elastic deformation field
    e = eps(u)
    stress_oldphi = as_tensor(C[i,j,k,l]*e[k,l], (i,j))
    dWdphi = as_tensor( dCdphi[i,j,k,l,m]*e[i,j]*e[k,l], (m,) )


    F_PF  = dot((phi-phi_n)/dt, q)*dx + \
            L*dot(dFdphi,q)*dx + \
            0.5*L*dot(dWdphi,q)*dx + \
            kappa*L*inner(grad(phi), grad(q))*dx


    a = derivative(F_PF, phi, dq)
    problem = PhaseField(a, F_PF)
    solver.solve(problem, phi.vector())
    phi_n.assign(phi)


    if inc < 50:
        if inc % 10 < 1e-4:
            
            file_phi.write(phi, np.round(t,1))
            
        else:
            if inc % 10 < 1 + 1e-4:
                
                W_elas = project(0.5*C[i,j,k,l]*e[i,j]*e[k,l], S)
                W_elas.rename("W_elas", "W_elas")
                file_W.write(W_elas, np.round(t-0.1,1))
                
                grad_u = project(grad(u),T)
                grad_u.rename("grad_u","grad_u")
                file_uij.write(grad_u, np.round(t-0.1,1))
                
                stress = project(stress_oldphi,T)
                stress.rename("stress","stress")
                file_sigma.write(stress, np.round(t-0.1,1))
                
    else:
        if inc % 5 < 1e-4:
            
            file_phi.write(phi, np.round(t,1))

        if inc % 5 < 1 + 1e-4 and inc % 5 > 1e-4:
            
            W_elas = project(0.5*C[i,j,k,l]*e[i,j]*e[k,l], S)
            W_elas.rename("W_elas", "W_elas")
            file_W.write(W_elas, np.round(t-0.1,1))
            
            grad_u = project(grad(u),T)
            grad_u.rename("grad_u","grad_u")
            file_uij.write(grad_u, np.round(t-0.1,1))
            
            stress = project(stress_oldphi,T)
            stress.rename("stress","stress")
            file_sigma.write(stress, np.round(t-0.1,1))

    t = t+dt

file_phi.close()
file_W.close()
file_sigma.close()
file_uij.close()