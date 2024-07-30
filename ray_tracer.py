import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))

                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def sphere_intersect(sphere, ray_origin, ray_direction):
    L = ray_origin - sphere.position
    b = 2.0 * np.dot(ray_direction, L)
    c = np.dot(L, L) - sphere.radius ** 2
    delta = b ** 2 - 4 * c
    t = 0
    if delta > 0:
        sqrt_delta = np.sqrt(delta)
        t1 = (-b + sqrt_delta) / 2
        t2 = (-b - sqrt_delta) / 2
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None, None, None  # Both t1 and t2 are negative

        intersection_point = ray_origin + t * np.array(ray_direction)
        normal_vector = (intersection_point - sphere.position) / sphere.radius
        return t, intersection_point, normal_vector
    return None, None, None


def box_intersect(cube,ray_origin, ray_direction):
    # Calculate the half-size of the box
    half_size = cube.scale / 2

    # Initialize t_min and t_max to represent the interval of intersection
    t_min = ((cube.position - half_size)- ray_origin) / ray_direction
    t_max = ((cube.position - half_size)- ray_origin) / ray_direction

    # Ensure t_min is the entry and t_max is the exit for each slab
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    # Find the largest t1 and smallest t2
    t_enter = np.max(t1)
    t_exit = np.min(t2)

    # If the intervals overlap, there is an intersection
    # if t_enter <= t_exit:
    #     # Calculate the intersection point

    if t_enter < t_exit and t_exit > 0:
        Intersection_Point= ray_origin + t_enter * ray_direction
        normal_vector = np.zeros(3)
        # Determine the normal by checking which t_min or t_max was used
        if t_enter == t1[0]:
            normal_vector= np.array([-1, 0, 0]) if ray_direction[0] < 0 else np.array([1, 0, 0])
        elif t_enter == t1[1]:
            normal_vector= np.array([0, -1, 0]) if ray_direction[1] < 0 else np.array([0, 1, 0])
        elif t_enter == t1[2]:
            normal_vector= np.array([0, 0, -1]) if ray_direction[2] < 0 else np.array([0, 0, 1])
        return t_enter, Intersection_Point, normal_vector
    return None, None, None


def intersect_objects(objects, ray_origin, ray_direction):
    min_distance = float('inf')
    nearest_object = None
    nearest_intersection_point = None
    nearest_normal_vector = None
    nearest_idx_material= None

    for obj in objects:
        if isinstance(obj, Sphere):
            distance, Intersection_Point, Normal_vector = sphere_intersect(obj, ray_origin, ray_direction)
        elif isinstance(obj, InfinitePlane):
            continue
            distance, Intersection_Point, Normal_vector = plane_intersect(obj,ray_origin, ray_direction)
        elif isinstance(obj, Cube):
            distance, Intersection_Point, Normal_vector= box_intersect(obj, ray_origin, ray_direction)
        else:
            continue

        if distance is not None and distance < min_distance:
            min_distance =distance
            nearest_object = obj
            nearest_intersection_point = Intersection_Point
            nearest_normal_vector = Normal_vector
            nearest_idx_material = obj.material_index

    return nearest_object, min_distance, nearest_intersection_point, nearest_normal_vector,nearest_idx_material


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def refract(ray_direction, normal, eta1, eta2):
    cos_i = -np.dot(normal, ray_direction)
    sin_t2 = (eta1 / eta2) ** 2 * (1 - cos_i ** 2)

    if sin_t2 > 1:
        return None  # Total internal reflection

    cos_t = np.sqrt(1 - sin_t2)
    refraction_ratio = eta1 / eta2

    return refraction_ratio * ray_direction + (refraction_ratio * cos_i - cos_t) * normal

def get_color(intersection_point, ray_direction, nearest_distance, nearest_normal, nearest_material, lights, surfaces, depth,scene_settings):
    color = np.zeros(3)

    view_dir = -ray_direction


    for light in lights:
        # Shadow check.Offset shadow ray origin to avoid self-intersection issues
        shifted_point = intersection_point + nearest_normal * 1e-5
        intersection_to_light = light.get_direction(shifted_point)
        _,min_distance, _ ,_ ,_= intersect_objects(surfaces,shifted_point, intersection_to_light)
        intersection_to_light_distance= light.get_euclidean_distance(intersection_point)
        if min_distance is not None and min_distance < intersection_to_light_distance:
            in_shadow = True
        else:
            in_shadow = False

        if not in_shadow:

            illumination = np.zeros((3))

            # Diffuse component
            diffuse_intensity = max(0, np.dot(nearest_normal, intersection_to_light))
            illumination += nearest_material.diffuse_color * light.color * diffuse_intensity

            # Specular component using Phong model
            reflect_dir = reflected(intersection_to_light, nearest_normal)    #compute reflection direction - or +
            specular_intensity = max(0, np.dot(reflect_dir, view_dir))
            specular = nearest_material.specular_color * (light.color * light.specular_intensity) * (specular_intensity ** nearest_material.shininess)
            illumination += specular

            color += illumination
        else:
            # Apply shadow intensity
            shadow_factor = 1 - light.shadow_intensity
            color += (nearest_material.diffuse_color * light.color * shadow_factor) #check this!!

    # return np.clip(color, 0, 1)

    reflection_color = np.zeros(3)
    # if np.any(nearest_material.reflection_color > 0) and depth < scene_settings.max_recursions:
    #     reflection_ray_direction = reflect(ray_direction, nearest_normal)
    #     # Not good at all!!!
    #     # reflection_t, reflection_point, reflection_normal, reflection_material,nearest_index_material = intersect_objects(surfaces,intersection_point,reflection_ray_direction)
    #     # reflection_color = get_color(intersection_point, reflection_ray_direction, reflection_t, reflection_point,reflection_normal, reflection_material, lights, scene_settings, objects, depth + 1)
    #     # reflection_color *= nearest_material.reflection_color
    #
    # background_color = scene_settings.background_color
    # if nearest_material.transparency > 0 and depth < scene_settings.max_recursions:
    #     refraction_ray_direction = refract(ray_direction, nearest_normal, 1, nearest_material.transparency)
    #     if refraction_ray_direction is not None:
    #         refraction_t, refraction_point, refraction_normal, refraction_material,nearest_index_material = intersect_objects(surfaces,intersection_point,refraction_ray_direction)
    #         background_color += get_color(intersection_point, refraction_ray_direction, refraction_t, refraction_point, refraction_normal, refraction_material, lights, scene_settings, objects,depth + 1)

    # Combine components
    # final_color = (scene_settings.background_color * nearest_material.transparency) + (color * (1 - nearest_material.transparency)) + reflection_color
    return np.clip(color, 0, 1)

def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array * 255))
    image.save("rendered_image.png")




def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, nargs='?', default='scenes/pool.txt', help='Path to the scene file')
    # parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()
    Lights,Materials, Surfaces= [], [],[]
    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    for obj in objects:
        if isinstance(obj, Light):
            Lights.append(obj)
        if isinstance(obj, Material):
            Materials.append(obj)
        if isinstance(obj, (Sphere, InfinitePlane, Cube)):
            Surfaces.append(obj)
        else:
            continue
        


    width, height = args.width, args.height

    image_array = np.zeros((height, width, 3))

    camera.update_screen_ratio(width, height)
    half_width, half_height = width / 2, height / 2
    for j in range(height):
        for i in range(width):
            # Calculate the position of the pixel in world coordinates
            pixel_location = camera.screen_location(i, j, half_width, half_height)
            # Calculate the direction of the ray from the camera position to the pixel position
            ray_direction = camera.ray_direction(pixel_location)

            # Check for intersections with objects
            nearest_object, Distance_to_intersection, nearest_intersection_point, nearest_normal, nearest_index_material = intersect_objects(objects, camera.position, ray_direction)

            if nearest_object is not None:
                # Calculate the color of the pixel based on the intersection
                color = get_color(nearest_intersection_point, ray_direction, Distance_to_intersection, nearest_normal,
                                  Materials[nearest_index_material], Lights, Surfaces, 0, scene_settings)

                image_array[j, i] = color


    save_image(image_array)

if __name__ == '__main__':
    main()
