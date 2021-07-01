from numpy.core.defchararray import zfill
import taichi as ti
import numpy as np
from .camera import *
from .shading import *
from .renderer_utils import ray_aabb_intersection, intersect_sphere, ray_plane_intersect, reflect, refract

inf = 1e8
eps = 1e-4

@ti.data_oriented
class ParticleRenderer:

    padding = 3 # extra padding to avoid cropping some of the projected sphere

    def __init__(self, system, radius=0.025, main_res=512):
        self.system = system
        system.renderer = self
        self.main_res = main_res
        self.radius = radius
        self.epsilon = 20.0 * self.radius

        ''' directional light '''
        self.camera_main = Camera(res=(main_res, main_res), pos=[0, 0.5, 2.5], target=[0, 0, 0])
        self.camera_main.add_buffer("pos", dim=3, dtype=float)
        self.camera_main.add_buffer("zbuf", dim=0, dtype=float)
        self.camera_main.add_buffer("normal", dim=3, dtype=float)
        self.main_img = self.camera_main.img

        light_y_pos = 2.0 - eps
        light_x_min_pos = -0.15
        light_x_range = 0.3
        light_z_min_pos = 1.0
        light_z_range = 0.3
        self.light_area = light_x_range * light_z_range
        self.light_vertices = [
            ti.Vector([light_x_min_pos, light_y_pos, light_z_min_pos]),
            ti.Vector([light_x_min_pos, light_y_pos, light_z_min_pos + light_z_range]),
            ti.Vector([light_x_min_pos + light_x_range, light_y_pos, light_z_min_pos + light_z_range]),
            ti.Vector([light_x_min_pos + light_x_range, light_y_pos, light_z_min_pos]),
        ]
        
        self.left_wall = [ti.Vector([-1.1, 0.0, 0.0]), ti.Vector([-1.1, 0.0, 2.0]), ti.Vector([-1.1, 2.0, 2.0]), ti.Vector([-1.1, 2.0, 0.0])]
        self.color_left = ti.Vector([0.65, 0.05, 0.05])
        self.right_wall = [ti.Vector([1.1, 0.0, 0.0]), ti.Vector([1.1, 2.0, 0.0]), ti.Vector([1.1, 2.0, 2.0]), ti.Vector([1.1, 0.0, 2.0])]
        self.color_right = ti.Vector([0.12, 0.45, 0.15])

        self.light_min_pos = self.light_vertices[0]
        self.light_max_pos = self.light_vertices[2]
        self.light_normal = ti.Vector([0.0, -1.0, 0.0])
        self.light_color = ti.Vector([0.9, 0.85, 0.7])
        self.light_intensity = 200

        self.camera_shadow = Camera(res=(2048, 2048), mainimg=False,
            pos=[light_x_min_pos + light_x_range / 2, light_y_pos + light_x_range / 2, light_z_min_pos + light_z_range / 2],
            target=[light_x_min_pos + light_x_range / 2, 0.0, light_z_min_pos + light_z_range / 2],
            up=[0, 0, 1],
            fov=45)
        self.camera_shadow.add_buffer("zbuf", dim=0, dtype=float)

    '''
    Clear camera
    '''
    @ti.kernel
    def clear_camera(self, camera: ti.template()):
        for I in ti.grouped(camera.img):
            camera.zbuf[I] = 0
            camera.img[I].fill(0)
            camera.normal[I].fill(0)
            camera.pos[I].fill(0)

    '''
    Calculates G-buffer
    '''
    @ti.kernel
    def calculate_buffers(self, camera: ti.template()):
        camera.W2V[None] = camera.L2W[None].inverse()
        # first pass: visibility splatting
        for i in range(self.system.num_particles_max):
            if i >= self.system.num_particles[None]:
                continue
            # particle center coordinate transfer
            # particle position view space 4d homogeneous coord [x, y, z, 1]
            pos_view = ti.Vector.zero(float, 3)
            pos_view = xyz(camera.W2V @ position(self.system.pos[i]))
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
            if pos_view.z > 0 and 0 <= xmin < xmax < camera.res[0] and 0 <= ymin < ymax < camera.res[1]:
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
                            z = camera.depth(depth)
                            # overwrite if closer
                            if z >= ti.atomic_max(camera.zbuf[row, column], z):
                                if ti.static(hasattr(camera, "normal")):
                                    frag_surface = ti.Vector([frag_view_space[0], frag_view_space[1], depth])
                                    normal = (frag_surface - pos_view).normalized()
                                    normal_world = xyz(camera.L2W @ direction(normal))
                                    pos_world = xyz(camera.L2W @ position(frag_surface))
                                    camera.img[row, column] = self.system.col[i] # diffuse
                                    camera.normal[row, column] = normal_world
                                    camera.pos[row, column] = pos_world


    @ti.func
    def intersect_light(self, pos, d, tmax):
        hit, t, _ = ray_aabb_intersection(self.light_min_pos, self.light_max_pos, pos, d)
        if hit and 0 < t < tmax:
            hit = 1
        else:
            hit = 0
            t = inf
        return hit, t

    '''
    Wall intersection from Cornell Box example
    '''
    @ti.func
    def intersect_scene(self, pos, ray_dir):
        closest, normal = inf, ti.Vector.zero(ti.f32, 3)
        c = ti.Vector.zero(ti.f32, 3)

        # left
        pnorm = ti.Vector([1.0, 0.0, 0.0])
        cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([-1.1, 0.0,
                                                                0.0]), pnorm)
        if 0 < cur_dist < closest:
            closest = cur_dist
            normal = pnorm
            c = self.color_left
        # right
        pnorm = ti.Vector([-1.0, 0.0, 0.0])
        cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([1.1, 0.0, 0.0]),
                                        pnorm)
        if 0 < cur_dist < closest:
            closest = cur_dist
            normal = pnorm
            c = self.color_right
        # bottom
        gray = ti.Vector([0.93, 0.93, 0.93])
        pnorm = ti.Vector([0.0, 1.0, 0.0])
        cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]),
                                        pnorm)
        if 0 < cur_dist < closest:
            closest = cur_dist
            normal = pnorm
            c = gray
        # top
        pnorm = ti.Vector([0.0, -1.0, 0.0])
        cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 2.0, 0.0]),
                                        pnorm)
        if 0 < cur_dist < closest:
            closest = cur_dist
            normal = pnorm
            c = gray
        # far
        pnorm = ti.Vector([0.0, 0.0, 1.0])
        cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]),
                                        pnorm)
        if 0 < cur_dist < closest:
            closest = cur_dist
            normal = pnorm
            c = gray
        # light
        
        hit_l, cur_dist = self.intersect_light(pos, ray_dir, closest)
        if hit_l and 0 < cur_dist < closest:
            # technically speaking, no need to check the second term
            closest = cur_dist
            normal = self.light_normal
            c = self.light_color

        return closest, normal, c


    '''
    Shadow map functions
    '''
    @ti.func
    def shadowmap_soft(self, pos):
        bias = eps
        light_size = 16
        n_sample = 64
        n_ring = 10
        radius = 1 / n_sample
        radius_step = radius
        angle = ti.random() * 2 * math.pi
        angle_step = 2 * math.pi * n_ring / n_sample

        pos_shadow = xyz(self.camera_shadow.W2V @ position(pos))
        zbuf_UV = self.camera_shadow.uncook(pos_shadow)
        z_shadow = self.camera_shadow.depth(pos_shadow.z)
        visibility = 0.0
        for _ in range(n_sample):
            delta_UV = ti.Vector([ti.cos(angle), ti.sin(angle)]) * (radius ** 0.75) * light_size
            angle += angle_step
            radius += radius_step
            #print(zbuf_UV, delta_UV)
            shadow_depth = texture(self.camera_shadow.zbuf, zbuf_UV + delta_UV)
            if 0 <= shadow_depth < z_shadow - bias:
                visibility += 1.0
        return visibility / n_sample



    @ti.func
    def shadowmap(self, pos):
        pos_shadow = xyz(self.camera_shadow.W2V @ position(pos))
        zbuf_UV = self.camera_shadow.uncook(pos_shadow)
        z_shadow = self.camera_shadow.depth(pos_shadow.z)
        bias = eps
        visibility = 1.0
        if texture(self.camera_shadow.zbuf, zbuf_UV) > z_shadow + bias:
            visibility = 0.0
        return visibility

    @ti.func
    def ssao(self, pos):
        ao_radius = self.radius * 15
        n_sample = 64
        sample = 0
        visible = 0.0
        while sample < n_sample:
            rand_vec = ti.Vector([ti.random(), ti.random(), ti.random()]) * 2 - 1.0
            if (rand_vec ** 2).sum() <= 1.0:
                sample += 1
                pos_test = pos + rand_vec * ao_radius
                pos_test_view = xyz(self.camera_main.W2V @ position(pos_test))
                pos_test_UV = self.camera_main.uncook(pos_test_view)
                z_test = self.camera_main.depth(pos_test_view.z)
                if z_test >= texture(self.camera_main.zbuf, pos_test_UV):
                    visible += 1.0
        return min(1.0, visible / n_sample * 2)




    '''
    Shading
    '''
    @ti.kernel
    def shade_particles(self):
        camera = self.camera_main
        # third pass: shading
        for I in ti.grouped(camera.img):
            rayorig, viewdir = camera.pixel_ray(I)
            closest, normal, color = self.intersect_scene(rayorig, viewdir)
            pos_world = rayorig + viewdir * closest
            pos_view = xyz(camera.W2V @ position(pos_world))
            z = camera.depth(pos_view.z)
            if z < camera.zbuf[I]:
                normal = camera.normal[I]
                color = camera.img[I]
                pos_world = camera.pos[I]

            # ambient
            ao = self.ssao(pos_world)
            color = color * 0.2 * ao
            # diffuse shadowed
            visibility = self.shadowmap_soft(pos_world)
            color += visibility * shade_area_diffuse(pos_world, normal, color, 
                    -self.light_normal, self.light_vertices, self.light_color, self.light_intensity)
            color += shade_area_diffuse(pos_world, normal, color, 
                    ti.Vector([1.0, 0.0, 0.0]), self.left_wall, self.color_left, self.light_intensity * 0.02)
            color += shade_area_diffuse(pos_world, normal, color, 
                    ti.Vector([-1.0, 0.0, 0.0]), self.right_wall, self.color_right, self.light_intensity * 0.02)
            
            #camera.img[I] = ti.Vector([1.0, 1.0, 1.0]) * ao * visibility
            # reflection 
            #refldir = viewdir - 2 * viewdir.dot(normal) * normal

            # tone mapping
            #camera.img[I] = camera.img[I] * 1.6 / (1.0 + camera.img[I])
            # gamma correction
            camera.img[I] = color ** (1 / 2.2)
   
    def render_main(self):
        self.clear_camera(self.camera_main)
        self.camera_shadow.zbuf.fill(0)
        self.calculate_buffers(self.camera_shadow)
        self.calculate_buffers(self.camera_main)
        self.shade_particles()

    '''
    Main render function which renders to the GUI.
    '''
    def render(self, gui):
        gui.clear()
        self.camera_main.from_mouse(gui)
        self.render_main()
        gui.set_image(self.main_img)
        #gui.set_image(self.camera_shadow.zbuf)

