'''
FEM research project series:
    1. Deformation gradient F
    2. Calculate strain energy Phi(F) with corotated linear elasticity model 
    3. Force

'''
import taichi as ti
import math

import util

ti.init(arch=ti.gpu)

# procedurally setting up the cantilever
center_x, center_y = 0.6, 0.5

N = 7
N_triangles = 6


# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
total_energy = ti.field(ti.f32, (), needs_grad=True)
grad = ti.Vector.field(2, ti.f32, N)  # force on vertices
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles) # rest volume/area


deformation_gradient = ti.Matrix.field(2, 2, ti.f32, ())
strain_engergy = ti.field(ti.f32, ())


# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)

# time-step size (for simulation, 16.7ms)
h = 0.8e-3
# substepping
substepping = 20
# time-step size (for time integration)
dh = h/substepping

curser_radius = 0.05

curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())

# rendering
width = 800
height = 800
pixels = ti.Vector.field(3, ti.f32, shape=(width, height))




# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    for i in range(N_triangles):
        triangles[i] = ti.Vector([6, i, (i+1)%N_triangles])



@ti.kernel
def initialize():
    YoungsModulus[None] = 1e5
    PoissonsRatio[None] = 0.2
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))
    for i in range(N_triangles):
        x[i] = ti.Vector([center_x + 0.2*ti.cos(i*math.pi*2/6), center_y + 0.2*ti.sin(i*math.pi*2/6)])
        v[i] = ti.Vector([0.0, 0.0])
    x[6] = ti.Vector([center_x, center_y])
    v[6] = ti.Vector([0.0, 0.0])

@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R, S

@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2


@ti.func
def compute_F(i):
    Ds = compute_D(i)
    F = Ds@elements_Dm_inv[i]
    return F

@ti.func
def frobenius_norm(M):
    acc = 0.0
    for i, j in ti.ndrange(2, 2):
        acc += M[i,j] ** 2
    return ti.sqrt(acc)

@ti.func
def strain_energy(i):
    Ds = compute_D(i)
    F = Ds@elements_Dm_inv[i] # Equation (4.5)

    R, S = compute_R_2D(F)
    Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
    Phi = LameMu[None]* frobenius_norm(F-R) +  LameLa[None]/2 * (((R.transpose())@F-Eye).trace())**2 # Equation (3.4)
    energy = elements_V0[i] * Phi # Equation (4.6)
    return energy

@ti.kernel
def compute_force():
    for i in grad:
        grad[i] = ti.Vector([0, 0])
    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity FEM part 1, page 19
        R, S = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose()) # Equation (4.7)
        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc   

@ti.kernel
def update():
    deformation_gradient[None] = compute_F(0)
    strain_engergy[None] = strain_energy(0)

    for i in range(N):
        acc = -grad[i]/m #- ti.Vector([0.0, g])
        v[i] += dh*acc
        x[i] += dh*v[i]

    # explicit damping (ether drag)
    for i in v:
        v[i] *= ti.exp(-dh*5)

    # mouse control
    for i in range(N):
        if picking[None]:      
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    # boundary
    for i in range(N):
        if x[i][0] <= 0 or x[i][0] >= 1:
            v[i][0] = -0.9*v[i][0]
        if x[i][1] <= 0 or x[i][1] >= 1:
            v[i][1] = -0.9*v[i][1]

# -------------------- triangle rendering -----------------------
@ti.dataclass
class Triangle:
    p1: ti.math.vec2
    p2: ti.math.vec2
    p3: ti.math.vec2

    @ ti.func
    def mask(self, p):
        alpha = ((self.p2.y - self.p3.y)*(p.x - self.p3.x) + (self.p3.x - self.p2.x)*(p.y - self.p3.y)) / \
            ((self.p2.y - self.p3.y)*(self.p1.x - self.p3.x) + (self.p3.x - self.p2.x)*(self.p1.y - self.p3.y))
        beta = ((self.p3.y - self.p1.y)*(p.x - self.p3.x) + (self.p1.x - self.p3.x)*(p.y - self.p3.y)) / \
            ((self.p2.y - self.p3.y)*(self.p1.x - self.p3.x) + (self.p3.x - self.p2.x)*(self.p1.y - self.p3.y))
        gamma = 1.0 - alpha - beta

        t = 0.0

        t_alpha = util.smoothstep(-0.002, 0.002, alpha)
        t_beta = util.smoothstep(-0.002, 0.002, beta)
        t_gamma = util.smoothstep(-0.002, 0.002, gamma)

        t = ti.min(ti.min(t_alpha, t_beta), t_gamma)

        return t
    
@ti.kernel
def render(width: ti.i32, height: ti.i32):
    # draw something on your canvas
    paint_color = ti.Vector([0.93, 0.19, 0.49]) 
    for i,j in pixels:
        coords_x = i / width
        coords_y = j / height
        color = ti.Vector([0.0, 0.0, 0.0]) # init your canvas to black

        mask = 0.0
        for  tidx in range(N_triangles):
            t = Triangle(x[triangles[tidx][0]], x[triangles[tidx][1]], x[triangles[tidx][2]])
            mask = ti.max(mask, t.mask(ti.Vector([coords_x, coords_y])))

        color = paint_color*mask

        pixels[i,j] = color

# init once and for all
meshing()
initialize()
initialize_elements()

window = ti.ui.Window("Title", (800, 800), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
while window.running:
    picking[None]=0
    window.get_event()
    if window.is_pressed(ti.ui.LMB):
  
        curser[None] = window.get_cursor_pos()
        picking[None] = 1

    for i in range(substepping):
        compute_force()
        update()

    value = 0
    color = (1.0, 1.0, 1.0)
    with gui.sub_window("Deformation Gradient", x=0, y=0, width=0.25, height=0.1):
        gui.text(f'| {deformation_gradient[None][0,0]:.2f}  ' + f'{deformation_gradient[None][0,1]:.2f}')
        gui.text(f'| {deformation_gradient[None][1,0]:.2f}  ' + f'{deformation_gradient[None][1,1]:.2f}')
    
    with gui.sub_window("Strain Energy", x=0, y=0.11, width=0.25, height=0.1):
        gui.text(f'{strain_engergy[None]:.2f}')

    
    render(width, height)

    canvas.set_image(pixels)
    window.show()