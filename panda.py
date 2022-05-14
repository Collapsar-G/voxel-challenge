import os; os.environ["TI_VISIBLE_DEVICE"] = "1"
from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0, exposure=3)
scene.set_floor(-0.9, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.01, (185/256,222/256,201/256))
black = vec3(0.05, 0.05, 0.05)
white = vec3(0.9, 0.9, 0.9)
@ti.func
def initialize_wall(x, y, z, r, g, b, pose_x, pose_y, pose_z):
    for i, j, k in ti.ndrange((-x, x), (-y, y), (-z, z)):
        scene.set_voxel(vec3(i + pose_x, j + pose_y, k + pose_z), 1, vec3(r, g, b))
@ti.func
def rotate3d(p, axis, ang):
    ca, sa = ti.cos(ang), ti.sin(ang)
    return mix(dot(p, axis) * axis, p, ca) + cross(axis, p) * sa
@ti.func
def initialize_ellipsoid(x, y, z, color, pose, axis1, ang1, axis2, ang2, axis3, ang3):
    for i, j, k in ti.ndrange((-x, x), (-y, y), (-z, z)):
        if (i / x) ** 2 + (j / y) ** 2 + (k / z) ** 2 <= 1:
            temp = pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)), axis3,
                                   radians(ang3))
            if -64 < temp.x < 64 and -64 < temp.y < 64 and -63 < temp.z < 63:
                scene.set_voxel(temp, 1, color)
@ti.func
def initialize_eyes(x, y, z, color, pose, axis1, ang1, axis2, ang2, axis3, ang3, location, r):
    for i, j, k in ti.ndrange((-x, x), (-y, y), (-z, z)):
        coordinate = pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)),
                                     axis3, radians(ang3))
        if (i / x) ** 2 + (j / y) ** 2 + (k / z) ** 2 <= 1 and (coordinate - location).dot(
                coordinate - location) <= r ** 2:            scene.set_voxel(coordinate, 1, color)
@ti.func
def initialize_cylinder(r, h, color, pose, axis1, ang1, axis2, ang2, axis3, ang3):
    for i, j, k in ti.ndrange((-r, r), (-h, h), (-r, r)):
        if distance(vec2(i, k), vec2(0, 0)) <= r:
            scene.set_voxel(
                pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)), axis3,
                                radians(ang3)), 1, color)
@ti.func
def initialize_cylinder_half(r, h, color, pose, axis1, ang1, axis2, ang2, axis3, ang3):
    for i, j, k in ti.ndrange((-r, 0), (-h, h), (-r, r)):
        if distance(vec2(i, k), vec2(0, 0)) <= r:
            scene.set_voxel(
                pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)), axis3,
                                radians(ang3)), 1, color)
@ti.func
def initialize_mouth(r, h, color, pose, axis1, ang1, axis2, ang2, axis3, ang3, location, r_):
    for i, j, k in ti.ndrange((-r, r), (-h, h), (0, r)):
        coordinate = pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)),
                                     axis3, radians(ang3))
        if r - 1 <= distance(vec2(i, k), vec2(0, 0)) <= r and i <= 0 and (coordinate - location).dot(
                coordinate - location) <= r_ ** 2:
            scene.set_voxel(
                pose + rotate3d(rotate3d(rotate3d(vec3(i, j, k), axis1, radians(ang1)), axis2, radians(ang2)), axis3,
                                radians(ang3)), 1, color)
@ti.kernel
def main():
    #initialize_wall(64, 64, 10, 238 / 256, 140 / 256, 145 / 256, 0, 0, -53)
    #initialize_wall(1, 64, 64, 143 / 256, 217 / 256, 250 / 256, -63, 0, 0)
    initialize_ellipsoid(25, 20, 25, black, vec3(5, -11, 30), vec3(0, 0, 1), 0, vec3(0, 1, 0), 0, vec3(0, 0, 1), 0)
    initialize_ellipsoid(32, 28, 32, white, vec3(5, 4, 30), vec3(0, 0, 1), 0, vec3(0, 1, 0), 0, vec3(0, 0, 1), 0)
    initialize_ellipsoid(32, 32, 28, white, vec3(5, -36, 30), vec3(0, 0, 1), 0, vec3(0, 1, 0), 0, vec3(0, 0, 1), 0)
    initialize_cylinder(6, 2, black, vec3(27, 27, 30), vec3(0, 1, 0), 0, vec3(0, 0, 1), 0, vec3(1, 0, 0), 90)
    initialize_cylinder(6, 2, black, vec3(-17, 27, 30), vec3(0, 1, 0), 0, vec3(0, 0, 1), 0, vec3(1, 0, 0), 90)
    initialize_eyes(5, 5, 8, black, vec3(15, 10, 58), vec3(0, 1, 0), 90, vec3(0, 0, 1), -45, vec3(1, 0, 0), 0,
                    vec3(5, 4, 30), 33)
    initialize_eyes(5, 5, 8, black, vec3(-5, 10, 58), vec3(0, 1, 0), 90, vec3(0, 0, 1), 45, vec3(1, 0, 0), 0,
                    vec3(5, 4, 30), 33)
    initialize_cylinder_half(4, 10, black, vec3(5, 5, 53), vec3(0, 1, 0), 90, vec3(1, 0, 0), 90, vec3(0, 0, 1), 0)
    initialize_mouth(10, 64, black, vec3(5, 4, 0), vec3(0, 1, 0), 48, vec3(1, 0, 0), 90, vec3(0, 0, 1), 0,
                     vec3(5, 4, 30), 33)
    initialize_ellipsoid(15, 10, 25, black, vec3(25, -18, 30), vec3(1, 0, 0), 35, vec3(0, 1, 0), 0, vec3(0, 0, 1), 30)
    initialize_ellipsoid(15, 8, 8, black, vec3(25, -28, 45), vec3(0, 0, 1), 35, vec3(0, 1, 0), 50, vec3(1, 0, 0), 0)
    initialize_ellipsoid(8, 8, 8, black, vec3(29, -31, 47), vec3(0, 0, 1), 0, vec3(0, 1, 0), 0, vec3(1, 0, 0), 0)
    initialize_ellipsoid(15, 10, 25, black, vec3(-15, -18, 30), vec3(1, 0, 0), 35, vec3(0, 1, 0), 0, vec3(0, 0, 1), -30)
    initialize_ellipsoid(15, 8, 8, black, vec3(-15, -28, 45), vec3(0, 0, 1), -35, vec3(0, 1, 0), -50, vec3(1, 0, 0), 0)
    initialize_ellipsoid(8, 8, 8, black, vec3(-19, -31, 47), vec3(0, 0, 1), 0, vec3(0, 1, 0), 0, vec3(1, 0, 0), 0)
    initialize_ellipsoid(13, 13, 30, black, vec3(27, -49, 47), vec3(0, 0, 1), 0, vec3(0, 1, 0), 20, vec3(1, 0, 0), 0)
    initialize_ellipsoid(13, 13, 30, black, vec3(-17, -49, 47), vec3(0, 0, 1), 0, vec3(0, 1, 0), -20, vec3(1, 0, 0), 0)
main()
scene.finish()
