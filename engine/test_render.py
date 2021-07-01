import taichi as ti
import numpy as np
from fast_renderer import ParticleSplattingRenderer
from particle_io import ParticleIO

res = 1024

ti.init(arch=ti.cuda, default_fp=ti.f32)#, use_unified_memory=False, device_memory_fraction=0.7)

class System:
    pass

path = "../sim_2021-05-20_22-58-41/01100.npz"
np_x, np_v, np_color = ParticleIO.read_particles_3d(path)
num_part = len(np_x)
translate = np.array([-0.5, 0, 0.5]).reshape(1, -1)
print(num_part, np_x.shape, np_v.shape)

system = System()
renderer = ParticleSplattingRenderer(system, main_res=res, radius=0.001)
system.num_particles = ti.field(int, shape=())
system.num_particles_max = num_part
system.pos = ti.Vector.field(3, float, shape=(num_part,))
system.col = ti.Vector.field(3, float, shape=(num_part,))

system.col.from_numpy(np_color / 255.0) # populate colors before animation



gui = ti.GUI('Particle', (res, res))
frame = 500
while gui.running:
    path = "../sim_2021-05-20_22-58-41/%05d.npz" % frame
    np_x, np_v, np_color = ParticleIO.read_particles_3d(path)
    system.num_particles[None] = len(np_x)
    np_x = np.vstack([np_x, np.zeros((num_part - len(np_x), 3))])
    system.pos.from_numpy(np_x + translate)
    

    gui.get_event()
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    #print("Render")
    renderer.render(gui)
    gui.show("../frames/%05d.png" % frame)
    frame += 1
    if frame > 1100:
        gui.running = False
    
    