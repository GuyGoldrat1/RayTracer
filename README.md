# Python Ray Tracing Project

<img src="output/pool.png" alt="Pool" width="300"/>

## Overview

This project implements a simple ray tracer in Python, capable of rendering 3D scenes with basic geometric shapes like spheres, planes, and cubes. It supports features such as soft shadows, reflections, and material properties like diffuse, specular, and transparency. The ray tracer uses a scene file to define the objects, materials, lights, and camera settings.

## Features

- **3D Scene Rendering**: Renders 3D scenes with spheres, planes, and cubes.
- **Material Properties**: Supports diffuse and specular reflection, transparency, and reflection.
- **Soft Shadows**: Implements soft shadowing for more realistic lighting.
- **Custom Scene Files**: Define your scenes with a simple text file format.

## Project Structure

- `camera.py`: Defines the `Camera` class, handling camera position, orientation, and ray generation.
- `light.py`: Defines the `Light` class, managing light properties such as color, intensity, and position.
- `material.py`: Defines the `Material` class, representing the material properties of surfaces, including diffuse, specular, and transparency.
- `scene_settings.py`: Defines the `SceneSettings` class, which holds global scene settings such as background color and recursion depth.
- `surfaces/`: Directory containing classes for various surfaces:
  - `cube.py`: Implements the `Cube` class, representing a 3D cube object.
  - `infinite_plane.py`: Implements the `InfinitePlane` class, representing an infinite plane.
  - `sphere.py`: Implements the `Sphere` class, representing a 3D sphere object.
- `ray.py`: Defines the `Ray` class, which represents a ray in 3D space.

## Classes and Functions

### 1. **Camera**

Handles camera position, orientation, and ray generation.

### 2. **Light**

Manages light properties like color, intensity, position, and shadow intensity.

### 3. **Material**

Represents the material properties of surfaces, including:
- `diffuse_color`: The color of the object under diffuse lighting.
- `specular_color`: The color of the object under specular lighting.
- `transparency`: How transparent the material is.
- `shininess`: The shininess of the material, affecting specular highlights.
- `reflection_color`: The color reflected by the material.

### 4. **SceneSettings**

Holds global scene settings like background color and maximum recursion depth for reflections.

### 5. **Cube**

Represents a 3D cube object, including its position, size, and material properties.

### 6. **InfinitePlane**

Represents an infinite plane with a normal vector, material properties, and a constant distance from the origin.

### 7. **Sphere**

Represents a 3D sphere object, including its center, radius, and material properties.

### 8. **Ray**

Represents a ray in 3D space with a starting position and direction.

### Functions

- **parse_scene_file(file_path)**: Parses the scene file to create objects, lights, materials, and camera settings.
- **intersect_objects(objects, ray)**: Determines which object a ray intersects first and returns the object, distance, intersection point, normal vector, and material index.
- **reflected(vector, normal)**: Computes the reflection of a vector given a normal.
- **diffuse_and_specular(in_ray, nearest_object, nearest_normal, nearest_material, lights, surfaces, settings)**: Computes the local shading for a surface based on diffuse and specular reflections.
- **soft_shadows(ray, light_position, nearest_object, nearest_normal, surfaces, radius, N)**: Implements soft shadowing by sampling multiple points on the light source.
- **reflection_component(reflection_ray, settings, surfaces, materials, nearest_idx, lights, max_recu)**: Recursively computes the reflection color component for a given ray.
- **main()**: The main function that ties everything together, reads the scene file, sets up the camera and objects, and initiates the ray tracing process.

## How to Run

 Run the ray tracer with a scene file:
   ```bash
   python ray_tracer.py scenes/your_scene.txt output_image.png --width 250 --height 250
   ```

   - `scene_file`: Path to the scene file defining the scene.
   - `output_image`: Name of the output image file.
   - `--width`: Image width in pixels.
   - `--height`: Image height in pixels.

## Example Scene File

Here's an example of a simple scene file:

```
# Camera: 	px   	py   	pz 	lx  	ly  	lz 	ux  	uy  	uz 	sc_dist	sc_width
cam 	  	0   	10	-2 	0   	-100   	-4  	0   	1   	0  	1.4	1	
# Settings: 	bgr  	bgg  	bgb	sh_rays	rec_max 
set 		1  	1  	1   	5 	10

# Material:	dr    	dg    	db	sr   	sg   	sb 	rr   	rg  	rb	phong 	trans
mtl		0.95	0.07	0.07	1	1	1	0.2	0.1	0.1	30	0
mtl		0.95	0.07	0.95	1	1	1	0.1	0.2	0.1	30	0
mtl		0.07	0.07	0.95	1	1	1	0.1	0.1	0.2	30	0
mtl		0.9	0.9	0.9	0.5	0.5	0.5	0.05	0.05	0.05	30	0
mtl		0.1	0.1	0.1	1	1	1	0.1	0.1	0.1	30	0
mtl		0.95	0.95	0.07	1	1	1	0.2	0.2	0	30	0
mtl		0.3	0.8	0	0	0	0	0	0	0	1	0

# Plane:	nx	ny	nz	offset	mat_idx
pln		0	1	0	-1	7

# Spheres:	cx   	cy   	cz  	radius 	mat_idx
sph		-2	0	0	1	1
sph		0	0	0	1	2
sph		2	0	0	1	3
sph		-1	0	-2	1	4
sph		1	0	-2	1	5
sph		0	0	-4	1	6

# Lights:	px	py	pz	r	g	b	spec	shadow	width
lgt		0	3	0	0.5	0.5	0.3	1	0.9	1
lgt		-3	3	-3	0.5	0.5	0.3	1	0.9	1
lgt		-3	3	3	0.5	0.5	0.3	1	0.9	1
lgt		3	3	-3	0.5	0.5	0.3	1	0.9	1
lgt		3	3	3	0.5	0.5	0.3	1	0.9	1```

```
