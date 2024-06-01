import taichi as ti

@ti.func
def smoothstep(edge1, edge2, v):
    assert(edge1 != edge2)
    t = (v-edge1) / float(edge2-edge1)
    t = clamp(t, 0.0, 1.0)

    return (3-2 * t) * t**2

@ti.func
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)