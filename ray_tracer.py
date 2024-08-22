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
from ray import Ray

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



def intersect_objects(objects, ray):
    min_distance = float('inf')
    object_hit = None
    intersection_point = None
    normal_vector = None
    idx_material= None

    for obj in objects:
        if isinstance(obj, Sphere) or isinstance(obj, InfinitePlane) or isinstance(obj, Cube):
            distance, Intersection_Point, Normal_vector = obj.intersect(ray)

            if distance is not None and distance < min_distance:
                min_distance =distance
                object_hit = obj
                intersection_point = Intersection_Point
                normal_vector = Normal_vector
                idx_material = obj.material_index

    return object_hit, min_distance, intersection_point, normal_vector,idx_material



def reflected(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal


def diffuse_and_specular(in_ray ,nearest_object, nearest_normal, nearest_material, lights, surfaces, settings):
    local_shading = np.zeros((3))
    for light in lights:
        shifted_point = in_ray.position + nearest_normal * 0.001
        intersection_to_light = light.get_direction(shifted_point)
        light_intensity = soft_shadows(in_ray, light.position,nearest_object, nearest_normal, surfaces, light.radius, int(settings.root_number_shadow_rays))
        shadow_factor =  (1.0 - light.shadow_intensity) + light.shadow_intensity * light_intensity
        
        # Diffuse component
        diffuse_intensity = max(0, np.dot(nearest_normal, intersection_to_light))
        local_shading += shadow_factor * nearest_material.diffuse_color * light.color * diffuse_intensity

        # Specular component
        reflect_ray_direction = reflected(intersection_to_light, nearest_normal) 
        specular_intensity = max(0, np.dot(reflect_ray_direction, in_ray.direction)) 
        local_shading += shadow_factor*nearest_material.specular_color *(specular_intensity ** nearest_material.shininess) *(light.color * light.specular_intensity)

    return local_shading * np.array([1 - nearest_material.transparency]*3)

def soft_shadows(ray, light_position,nearest_object, nearest_normal, surfaces, radius, N):
    points = []
    ray_direction = ray.direction / np.linalg.norm(ray.direction)
    perp_vector1 = np.cross(ray_direction,[1,0,0])
    perp_vector1 = perp_vector1 / np.linalg.norm(perp_vector1)
    perp_vector2 = np.cross(ray_direction, perp_vector1)
    perp_vector2 = perp_vector2 / np.linalg.norm(perp_vector2)

    step_size = 2 * radius / N
    start_cell = light_position - radius* (perp_vector1 + perp_vector2)

    for i in range(N):
        for j in range(N):
            random_offset_i = np.random.uniform(0, step_size)
            random_offset_j = np.random.uniform(0, step_size)

            sampled_position = start_cell + (step_size *i* perp_vector1) + (step_size * j * perp_vector2) + (perp_vector1 * random_offset_i) + (perp_vector2 * random_offset_j)
            points.append(sampled_position)

    hits =[]
    shifted_point = ray.position + nearest_normal *0.001
    for light_point in points:
        intersection_to_light = (shifted_point - light_point)
        intersection_to_light = intersection_to_light / np.linalg.norm(intersection_to_light)
        ray = Ray(light_point,intersection_to_light)
        object ,min_distance, _ ,_ ,_= intersect_objects(surfaces, ray)
        if object != nearest_object:
            hits.append(0)
        else:
            hits.append(1)

    return sum(hits)/len(hits)



def reflection_component(reflection_ray, settings, surfaces, materials, nearest_idx, lights, max_recu):

    reflection_origin = reflection_ray.position + reflection_ray.direction * 0.001
    original_reflection = materials[nearest_idx].reflection_color
    if max_recu == 0:
        return original_reflection * settings.background_color 
    ray = Ray(reflection_origin,reflection_ray.direction)
    nearest_object, reflection_distance, reflection_point, reflection_normal, reflection_material_idx = intersect_objects(surfaces,ray)
    
    if nearest_object is not None:
        reflection_material = materials[reflection_material_idx]
        new_ray = Ray(reflection_point, reflection_ray.direction)

        reflection_color =  diffuse_and_specular(new_ray ,nearest_object, reflection_normal, reflection_material, lights, surfaces, settings)
    
        reflected_ray = Ray(reflection_point,reflected(reflection_ray.direction, reflection_normal))
        reflected_color = reflection_component(reflected_ray, settings, surfaces, materials, reflection_material_idx, lights, max_recu-1)
        reflection_color += reflected_color 
        
        return original_reflection * reflection_color 
    
    else:
        return original_reflection * settings.background_color
    



def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array * 255))
    image.save("output.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, nargs='?', default='scenes/pool.txt', help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=250, help='Image width')
    parser.add_argument('--height', type=int, default=250, help='Image height')
    args = parser.parse_args()
    Lights,Materials, Surfaces= [], [],[]

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

            #Prints 
            if ((j/height)*100)%(10) == 0 and i ==0:
                print(f'finished {(j/height)*100}%')

            pixel_location = camera.screen_location(i, j, half_width, half_height)

            cam_ray = Ray(camera.position,camera.ray_direction(pixel_location))
            color = np.zeros((3))
            background_color = np.zeros((3))
            
            nearest_object, distance_to_intersection, nearest_intersection_point, normal, nearest_index_material = intersect_objects(objects, cam_ray)
            if nearest_object is not None:
                material = Materials[nearest_index_material]

                inter_ray = Ray(nearest_intersection_point,cam_ray.direction)
                diffuse_and_specular_color = diffuse_and_specular(inter_ray ,nearest_object ,normal, material, Lights,Surfaces, scene_settings)

                #Handle transpercy
                if material.transparency > 0:
                        new_objects = [obj for obj in objects if obj != nearest_object]
                        new_nearest_object, distance_to_intersection, back_intersection_point, new_nearest_normal, new_index_material = intersect_objects(new_objects, cam_ray)
                        if new_nearest_object is not None:
                            back_material = Materials[new_index_material]
                            transperent_ray = Ray(back_intersection_point,cam_ray.direction)
                            background_color = diffuse_and_specular(transperent_ray, new_nearest_object ,new_nearest_normal, back_material, Lights,Surfaces, scene_settings)

                reflected_ray = Ray(nearest_intersection_point, reflected(cam_ray.direction,normal))
                reflection_color = reflection_component(reflected_ray, scene_settings,Surfaces, Materials ,nearest_index_material, Lights, scene_settings.max_recursions)

                color = (background_color * np.array([material.transparency]*3)) + diffuse_and_specular_color + reflection_color
                image_array[height - j - 1,  width -i - 1] = np.clip(color, 0, 1)  

            else:
                image_array[height - j - 1, width -i - 1] = scene_settings.background_color  

    save_image(image_array)

np.set_printoptions(precision=3, suppress=True)
if __name__ == '__main__':
    main()

