import taichi as ti
import numpy as np
from .camera import *
from .shading import *

@ti.data_oriented
class ParticleSplattingRenderer:

    padding = 3 # extra padding to avoid cropping some of the projected sphere

    def __init__(self, system, radius=0.025, main_res=512, 
        lightdir=[-1, -1, -4]):
        self.system = system
        system.renderer = self
        self.main_res = main_res
        self.radius = radius
        self.epsilon = 20.0 * self.radius

        ''' directional light '''
        self.camera_main = Camera(res=(main_res, main_res), pos=[1, 1, 2], target=[0.5, 0.5, 0.5])
        self.camera_main.add_buffer("zbuf", dim=0, dtype=float)
        self.camera_main.add_buffer("weight", dim=0, dtype=float)
        self.camera_main.add_buffer("normal", dim=3, dtype=float)
        self.main_img = self.camera_main.img

        self.light = ti.Vector(lightdir).normalized()

    '''
    Clear camera
    '''
    @ti.kernel
    def clear_camera(self, camera: ti.template()):
        for I in ti.grouped(camera.img):
            camera.zbuf[I] = float("inf")
            camera.weight[I] = 0
            camera.img[I].fill(0)
            camera.normal[I].fill(0)

    '''
    First pass of particle splatting
    '''
    @ti.kernel
    def calculate_surface(self, camera: ti.template()):
        nviews = 1
        if ti.static(isinstance(camera, CameraArray)):
            nviews = camera.n
            for i in ti.static(range(camera.n)):
                camera.W2V[i] = camera.L2W[i].inverse()
        else:
            camera.W2V[None] = camera.L2W[None].inverse()
        # first pass: visibility splatting
        for i in range(self.system.num_particles * nviews):
            # particle center coordinate transfer
            # particle position view space 4d homogeneous coord [x, y, z, 1]
            icam = i // self.system.num_particles
            ipart = i % self.system.num_particles
            pos_view = ti.Vector.zero(float, 3)
            if ti.static(isinstance(camera, CameraArray)):
                pos_view = xyz(camera.W2V[icam] @ position(self.system.pos[ipart]))
            else:
                pos_view = xyz(camera.W2V @ position(self.system.pos[ipart]))
            pos_img = camera.uncook(pos_view) # 2d image space position (x, y) in pixel unit
            # find the projected radius in image space
            ref_view_space = ti.Vector([pos_view[0] + self.radius, pos_view[1], pos_view[2]])
            ref_img_space = camera.uncook(ref_view_space)
            r_projected = abs(ref_img_space[0] - pos_img[0]) + self.padding # projected radius in pixel unit
            
            # fragment ranges to render
            xmin = int(min(max(0, pos_img[0] - r_projected), camera.res[0]))
            xmax = int(min(max(0, pos_img[0] + r_projected), camera.res[0]))
            ymin = int(min(max(0, pos_img[1] - r_projected), camera.res[1]))
            ymax = int(min(max(0, pos_img[1] + r_projected), camera.res[1]))
            if xmin <= xmax and ymin <= ymax:
                # process projected fragments and compute depth
                for row in range(xmin, xmax):
                    for column in range(ymin, ymax):
                        # discard fragment if its distance to particle center > projected radius
                        frag_view_space = ti.Vector([row, column, pos_view[2]]).cast(float)
                        frag_view_space = camera.cook(frag_view_space) # 3d position in view space
                        dis_projected = (frag_view_space - pos_view).norm()
                        if dis_projected <= self.radius:
                            # compute depth value for valid fragment
                            depth = pos_view[2] - ti.sqrt(self.radius ** 2 - dis_projected ** 2)
                            z = depth + self.epsilon # epsilon shifted depth
                            # overwrite if closer
                            if ti.static(isinstance(camera, CameraArray)):
                                if z < camera.zbuf[icam, row, column]:
                                    camera.zbuf[icam, row, column] = z
                            else:
                                ti.atomic_min(camera.zbuf[row, column], z)


    '''
    Second pass of particle splatting which calculates buffers
    Taped kernels must not have any global operations outside
    the parallelized region so nviews is passed as an argument.
    '''
    @ti.kernel
    def calculate_buffers(self, camera: ti.template(), nviews: ti.template()): # particle_position_field, particle_color_field
        # second pass: attribute blending
        for i in range(self.system.num_particles * nviews):
            icam = i // self.system.num_particles
            ipart = i % self.system.num_particles
            pos_view = ti.Vector.zero(float, 3)
            if ti.static(isinstance(camera, CameraArray)):
                pos_view = xyz(camera.W2V[icam] @ position(self.system.pos[ipart]))
            else:
                pos_view = xyz(camera.W2V @ position(self.system.pos[ipart]))
            pos_img = camera.uncook(pos_view) # 2d image space position (x, y) in pixel unit
            # find the projected radius in image space
            ref_view_space = ti.Vector([pos_view[0] + self.radius, pos_view[1], pos_view[2]])
            ref_img_space = camera.uncook(ref_view_space)
            r_projected = abs(ref_img_space[0] - pos_img[0]) + self.padding # projected radius in pixel unit

            # fragment ranges to render
            xmin = int(min(max(0, pos_img[0] - r_projected), camera.res[0]))
            xmax = int(min(max(0, pos_img[0] + r_projected), camera.res[0]))
            ymin = int(min(max(0, pos_img[1] - r_projected), camera.res[1]))
            ymax = int(min(max(0, pos_img[1] + r_projected), camera.res[1]))
            if xmin <= xmax or ymin <= ymax:
                # compute weights for each fragment
                for row in range(xmin, xmax):
                    for column in range(ymin, ymax):

                        frag_view_space = camera.cook(ti.Vector([row, column, pos_view[2]]).cast(float))  # 3d position in view space

                        dis_projected = (frag_view_space - pos_view).norm() # view space
                        dis_img_space = (ti.Vector([row, column]) - pos_img).norm()

                        if dis_img_space <= r_projected - self.padding:
                            w1 = 1 - dis_img_space / (r_projected - self.padding)

                            depth = (pos_view[2] - ti.sqrt(self.radius ** 2 - dis_projected ** 2))
                            # updates the depth of the fragment
                            frag_surface = ti.Vector([frag_view_space[0], frag_view_space[1], depth])
                            w2 = 0.0
                            if ti.static(isinstance(camera, CameraArray)):
                                w2 = (camera.zbuf[icam, row, column] - depth) / self.epsilon
                                #camera.zbuf[icam, row, column] = depth
                            else:
                                w2 = (camera.zbuf[row, column] - depth) / self.epsilon
                                #camera.zbuf[row, column] = depth
                            if w2 > 0:
                                weight = w1 * w2
                                normal = (frag_surface - pos_view).normalized()
                                if ti.static(isinstance(camera, CameraArray)):
                                    normal_world = xyz(camera.L2W[icam] @ direction(normal))
                                    camera.weight[icam, row, column] += weight # sum weights
                                    camera.img[icam, row, column] += weight * self.system.col[i] # blend color
                                    # transform normal to world space
                                    camera.normal[icam, row, column] += weight * normal_world # blend normal
                                else:
                                    normal_world = xyz(camera.L2W @ direction(normal))
                                    camera.weight[row, column] += weight # sum weights
                                    camera.img[row, column] += weight * self.system.col[i] # blend color
                                    # transform normal to world space
                                    camera.normal[row, column] += weight * normal_world # blend normal
    '''
    Shading
    '''
    @ti.kernel
    def shade_particles(self, camera: ti.template()):
        floor_height = 0.02
        # third pass: shading
        for I in ti.grouped(camera.img):
            rayorig, viewdir = camera.pixel_ray(I)
            if camera.weight[I] > 0:
                normal = camera.normal[I].normalized()
                color = camera.img[I] / camera.weight[I]
                camera.img[I] = shade_cooktorrance(
                    color, normal, -self.light, -viewdir)
                # reflection 
                refldir = viewdir - 2 * viewdir.dot(normal) * normal
                if refldir[1] > 0:
                    camera.img[I] += min(0.05, refldir[1] * 5) * sample_sky(rayorig, refldir)
                else:
                    fragpos = camera.uncook(ti.Vector([I.x, I.y, camera.zbuf[I]]).cast(float))
                    fragpos_w = xyz(camera.L2W[None] @ ti.Vector([fragpos.x, fragpos.y, camera.zbuf[I], 1]))
                    col, _ = sample_floor(floor_height, fragpos_w, viewdir, -viewdir)
                    camera.img[I] += min(0.05, -refldir[1] * 5) * col 
            else:
                # use simple raycast to render background
                camera.img[I] = sample_sky(rayorig, viewdir)
            if rayorig[1] > floor_height and viewdir[1] < 0:
                # shade floor
                col, intersect = sample_floor(floor_height, rayorig, viewdir, -self.light)
                # depth test
                depth = (camera.W2V @ position(intersect))[2] + self.epsilon
                if depth < camera.zbuf[I]:
                    camera.img[I] = col
            # tone mapping
            #camera.img[I] = camera.img[I] * 1.6 / (1.0 + camera.img[I])
            # gamma correction
            camera.img[I] = camera.img[I] ** (1 / 1.6)

    
    @ti.kernel
    def calc_color(self, camera: ti.template()):
        for I in ti.grouped(camera.img):
            if camera.weight[I] > 0:
                camera.img[I] = camera.img[I] / camera.weight[I]

    
    def render_main(self):
        self.clear_camera(self.camera_main)
        self.calculate_surface(self.camera_main)
        self.calculate_buffers(self.camera_main, 1)
        self.shade_particles(self.camera_main)

    '''
    Main render function which renders to the GUI.
    '''
    def render(self, gui):
        gui.clear()
        self.camera_main.from_mouse(gui)
        self.render_main()
        gui.set_image(self.main_img)

