import json
import numpy as np
import pyvista as pv

# ===========================
# 1. Загружаем данные
# ===========================
with open("world_cells_from_image_optimized.json", "r") as f:
    cells = json.load(f)

nx = max(c["i"] for c in cells) + 1
ny = max(c["j"] for c in cells) + 1

# ===========================
# 2. Преобразуем в XYZ
# ===========================
radius_earth = 6371.0  # Радиус Земли
points = []
colors = []

# Создаем массив, чтобы потом было удобно формировать треугольники
grid_points = np.zeros((nx, ny, 3))
grid_colors = np.zeros((nx, ny, 3), dtype=np.uint8)

for c in cells:
    i, j = c["i"], c["j"]
    theta = (i / (nx - 1)) * 2 * np.pi
    phi = np.pi/2 - (j / (ny - 1)) * np.pi
    r = radius_earth  # без рельефа, если нужен рельеф: r += c["elevation"]/1000

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    grid_points[i, j] = [x, y, z]
    grid_colors[i, j] = c["color"]

# ===========================
# 3. Создаем треугольники для сетки
# ===========================
faces = []
for i in range(nx - 1):
    for j in range(ny - 1):
        # 2 треугольника на каждый квадратик
        p0 = i * ny + j
        p1 = (i + 1) * ny + j
        p2 = (i + 1) * ny + (j + 1)
        p3 = i * ny + (j + 1)

        # треугольники
        faces.append([3, p0, p1, p2])
        faces.append([3, p0, p2, p3])

points_flat = grid_points.reshape(-1, 3)
colors_flat = grid_colors.reshape(-1, 3)
faces_flat = np.hstack(faces)

# ===========================
# 4. Создаем PyVista mesh
# ===========================
mesh = pv.PolyData(points_flat, faces_flat)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars=colors_flat, rgb=True)
plotter.add_axes()
plotter.show()
