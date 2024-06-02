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

# # procedurally setting up the cantilever
# center_x, center_y = 0.6, 0.5

# N = 7
# N_triangles = 6





# deformation_gradient = ti.Matrix.field(2, 2, ti.f32, ())
# strain_engergy = ti.field(ti.f32, ())


# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())



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
width = 1000
height = 1000
pixels = ti.Vector.field(3, ti.f32, shape=(width, height))

# World building
block_size = 50 # 50*50 block dimension
N_bx = width // block_size
N_by = height // block_size
building_block = ti.field(ti.i32, shape=(N_bx, N_by))

N_x = N_bx + 1 # Number of block in x-axis
N_y = N_by + 1 # Number of block in y-axis
N_grid_points = N_x * N_y

# geometric components
N_triangles = N_bx * N_by * 2 # 2 triangles per block
triangles = ti.Vector.field(3, ti.i32, N_triangles)


# simulation components
x = ti.Vector.field(2, ti.f32, N_grid_points)
v = ti.Vector.field(2, ti.f32, N_grid_points)
total_energy = ti.field(ti.f32, ())
grad = ti.Vector.field(2, ti.f32, N_grid_points)  # force on vertices
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles) # rest volume/area



# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    building_block[5,4] = 1
    building_block[6,4] = 1
    building_block[6,3] = 1
    for i, j in building_block:
        if building_block[i,j] == 1:
            bl = i * N_y + j # bottom left vertex index
            br = (i+1) * N_y + j # bottom right vertex index
            tl = i * N_y + (j+1) # top left vertex index
            tr = (i+1) * N_y + (j+1) # top right vertex index
            triangles[2*(i*N_by+j)] = ti.Vector([bl, br, tl])
            triangles[2*(i*N_by+j)+1] = ti.Vector([tl, br, tr])
    


@ti.kernel
def initialize():
    YoungsModulus[None] = 1e5
    PoissonsRatio[None] = 0.4
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))
    for i, j in ti.ndrange(N_x, N_y):
        x[i*N_y+j] = ti.Vector([block_size * i / width, block_size * j / height])
        v[i*N_y+j] = ti.Vector([0.0, 0.0])

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
print(triangles)
initialize()
initialize_elements()

window = ti.ui.Window("Title", (width, height), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
while window.running:
    picking[None]=0
    window.get_event()
    if window.is_pressed(ti.ui.LMB):
  
        curser[None] = window.get_cursor_pos()
        picking[None] = 1

    # for i in range(substepping):
    #     compute_force()
    #     update()

    # value = 0
    # color = (1.0, 1.0, 1.0)
    # with gui.sub_window("Deformation Gradient", x=0, y=0, width=0.25, height=0.1):
    #     gui.text(f'| {deformation_gradient[None][0,0]:.2f}  ' + f'{deformation_gradient[None][0,1]:.2f}')
    #     gui.text(f'| {deformation_gradient[None][1,0]:.2f}  ' + f'{deformation_gradient[None][1,1]:.2f}')
    
    # with gui.sub_window("Strain Energy", x=0, y=0.11, width=0.25, height=0.1):
    #     gui.text(f'{strain_engergy[None]:.2f}')

    
    render(width, height)

    canvas.set_image(pixels)
    window.show()