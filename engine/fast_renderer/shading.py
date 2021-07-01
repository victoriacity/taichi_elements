import taichi as ti
import math

EPS = 1e-4
roughness = 0.2
metallic = 0.0
specular = 0.04
ambient = 0.1

floor_color_a = ti.Vector([1.0, 1.0, 1.0])
floor_color_b = ti.Vector([0.6, 0.6, 0.6])
floor_delta = 0.1

@ti.func
def smoothstep(t): return t * t * (3 - 2 * t)
@ti.func
def clamp(x, l, h): return max(l, min(x, h))
@ti.func
def lerp(a, b, l): return a * (1 - l) + b * l

'''
Bilinear texture interpolation in pixel coordinates
'''
@ti.func
def texture(img, coord):
    x0 = int(coord.x)
    y0 = int(coord.y)
    dx = coord.x - x0
    dy = coord.y - y0
    v11 = 0.0
    if 0 <= x0 < img.shape[0] - 1 and 0 <= y0 < img.shape[1] - 1:
        v01 = lerp(img[x0, y0], img[x0 + 1, y0], dx)
        v10 = lerp(img[x0, y0 + 1], img[x0 + 1, y0 + 1], dx)
        v11 = lerp(v01, v10, dy)
    return v11


@ti.func
def brdf_cooktorrance(color, normal, lightdir, viewdir):
    halfway = (viewdir + lightdir).normalized()
    ndotv = max(viewdir.dot(normal), EPS)
    ndotl = max(lightdir.dot(normal), EPS)
    ndf = microfacet(normal, halfway)
    geom = geometry(ndotv, ndotl)
    f = fresnel(viewdir, halfway, color)
    ks = f
    kd = 1 - ks
    kd *= 1 - metallic
    diffuse = kd * color / math.pi
    specular = ndf * geom * f / (4 * ndotv * ndotl)
    return diffuse + specular

'''
Trowbridge-Reitz GGX microfacet distribution
'''
@ti.func
def microfacet(normal, halfway):
    alpha = roughness
    ggx = alpha ** 2 / math.pi
    ggx /= (normal.dot(halfway)**2 * (alpha**2 - 1.0) + 1.0) ** 2
    return ggx

'''
Fresnel-Schlick approximation
'''
@ti.func
def fresnel(view, halfway, color):
    specular_vec = ti.Vector([specular] * 3)
    f0 = specular_vec * (1 - metallic) + color * metallic
    hdotv = min(1, max(halfway.dot(view), 0))
    return f0 + (1.0 - f0) * (1.0 - hdotv) ** 5

'''
Smith's method with Schlick-GGX
'''
@ti.func
def geometry(ndotv, ndotl):
    k = (roughness + 1.0) ** 2 / 8
    geom = ndotv * ndotl\
        / (ndotv * (1.0 - k) + k) / (ndotl * (1.0 - k) + k)
    return max(0, geom)

@ti.func
def shade_cooktorrance(color, normal, lightdir, viewdir):
    costheta = max(0, normal.dot(lightdir))
    radiance = 2.0
    l_out = ambient * color
    if costheta > 0:
        l_out += brdf_cooktorrance(color, normal, lightdir, viewdir)\
                * costheta * radiance
    return l_out

@ti.func
def shade_simple(color, normal, lightdir, viewdir):
    fac = max(0, normal.dot(lightdir))
    diffuse = color * fac
    ambient = color * 0.1
    return diffuse + ambient

@ti.func
def shade_flat(color, normal, lightdir, viewdir):
    return color


@ti.func
def sample_sky(rayorig, raydir):
    l = max(0, raydir[1])
    return ti.Vector([0.1, 0.6, 0.95]) * (1 - l) ** 3 + ti.Vector([0.98, 0.98, 1]) * (1 - (1 - l) ** 3)

@ti.func
def sample_floor(floor_height, rayorig, raydir, light):
    raylength = (floor_height - rayorig[1]) / raydir[1]
    intersect = rayorig + raylength * raydir
    # checkerboard texture
    tex_idx = int(ti.floor(intersect[0] / floor_delta) + ti.floor(intersect[2] / floor_delta))
    floor_color = floor_color_b
    if tex_idx % 2 == 0:
        floor_color = floor_color_a
    col = shade_cooktorrance(floor_color, ti.Vector([0.0, 1.0, 0.0]), light, -raydir)
    return col, intersect

@ti.func
def form_factor_quad(pos, normal, verts):
    factor = 0.0
    for i in ti.static(range(4)):
        v = (verts[i] - pos).normalized()
        v_next = (verts[(i + 1) % 4] - pos).normalized()
        factor += normal.dot(v.cross(v_next).normalized()) \
            * ti.acos(v.dot(v_next))
    return max(0, factor / (2 * math.pi))


@ti.func
def shade_area_diffuse(pos, normal, color, light_dir, light_verts, light_color, light_intensity):
    diffuse = color * light_color * light_intensity \
        * form_factor_quad(pos, normal, light_verts)
    return diffuse

