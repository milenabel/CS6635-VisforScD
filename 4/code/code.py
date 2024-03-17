# Part 4: Euler’s Method[35 pts] 
# You are given some wind data as a 2D vector field of size 20x20. 
# Compute and visualize seed points and streamlines on this dataset using Euler's Method(en.wikipedia.org/wiki/Euler_method).  
# Data contains the wind data vectors (wind_vectors.raw) and also boilerplate python code (wind.py).  
# 1. Use random sampling to generate 15 seed points within the range [0,19] in both dimensions. 
#    Also, set the random seed manually to an arbitrary number. 
#    This will give you consistent seed points which will be helpful for comparison. Show an image of your plot. 
# 2. Trace a streamline from each point. Use a time step value of 0.3 and perform 8 steps for each streamline. 
#    You will need to write a function that will calculate the bilinear interpolation
#    (wikipedia.org/wiki/Bilinear_interpolation) of the 4 neighboring vectors. 
#    Bound your points to the data range [0,20] along both dimensions. 
#    You can stop tracing early if you reach the image boundary. Show an image of your plot. 
# 3. Trace streamlines again from the same seed points as the previous part. 
#    Make 3 more figures with the following parameters: 
#    • Step size 0.15, steps 16 
#    • Step size 0.075, steps 32 
#    • Step size 0.0375, steps 64 
# 4. Describe differences you see between the figures. 
#    Explain what divergence is. How does this relate to your results?(Tip:Read section 1.3.4 in the book) 

import numpy as np
import matplotlib.pyplot as plt

# Get data
vecs = np.reshape(np.fromfile("4/data/wind_vectors.raw"), (20,20,2))
vecs_flat = np.reshape(vecs, (400,2)) # useful for plotting
vecs = vecs.transpose(1,0,2) # needed otherwise vectors don't match with plot

# X and Y coordinates of points where each vector is in space
xx, yy = np.meshgrid(np.arange(0, 20),np.arange(0, 20))

# Plot vectors
plt.plot(xx, yy, marker='.', color='b', linestyle='none')
plt.quiver(xx, yy, vecs_flat[:,0], vecs_flat[:,1], width=0.001)
plt.savefig("4/figs/part4_initial.png")
plt.show()

# Set random seed for reproducibility
np.random.seed(42)

# Generate 15 random seed points
seed_points = np.random.uniform(low=0, high=19, size=(15, 2))

# Plot vectors and seed points
plt.figure(figsize=(10, 10))
plt.plot(xx, yy, marker='.', color='b', linestyle='none')
plt.quiver(xx, yy, vecs_flat[:, 0], vecs_flat[:, 1], width=0.001)
plt.scatter(seed_points[:, 0], seed_points[:, 1], color='r', zorder=5)
plt.xlim([0, 19])
plt.ylim([0, 19])
plt.title('Wind Vectors and Seed Points')
plt.savefig("4/figs/part4_1.png")
plt.show()


def bilinear_interpolation(x, y, vecs):
    """
    Perform bilinear interpolation for the vector at position (x, y).
    Args:
    - x: X coordinate
    - y: Y coordinate
    - vecs: 2D array of vectors

    Returns:
    - Interpolated vector at (x, y)
    """
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = int(np.ceil(x)), int(np.ceil(y))

    # Boundaries
    if x1 == x2: x2 += 1
    if y1 == y2: y2 += 1

    Q11 = vecs[y1, x1] if y1 < 20 and x1 < 20 else np.array([0, 0])
    Q12 = vecs[y2, x1] if y2 < 20 and x1 < 20 else np.array([0, 0])
    Q21 = vecs[y1, x2] if y1 < 20 and x2 < 20 else np.array([0, 0])
    Q22 = vecs[y2, x2] if y2 < 20 and x2 < 20 else np.array([0, 0])

    # Bilinear interpolation formula
    R1 = (x2 - x) / (x2 - x1) * Q11 + (x - x1) / (x2 - x1) * Q21
    R2 = (x2 - x) / (x2 - x1) * Q12 + (x - x1) / (x2 - x1) * Q22
    P = (y2 - y) / (y2 - y1) * R1 + (y - y1) / (y2 - y1) * R2

    return P

def trace_streamline(x, y, vecs, time_step, steps):
    """
    Trace a streamline from a seed point.
    Args:
    - x, y: Initial position
    - vecs: Vector field data
    - time_step: Time step value
    - steps: Number of steps to perform

    Returns:
    - Arrays of x and y coordinates of the streamline
    """
    xs, ys = [x], [y]
    for _ in range(steps):
        vx, vy = bilinear_interpolation(x, y, vecs)
        x, y = x + vx * time_step, y + vy * time_step
        # Boundary conditions
        if x < 0 or x >= 20 or y < 0 or y >= 20:
            break
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Parameters for tracing the streamline
time_step = 0.3
steps = 8

# Trace streamlines from each seed point
plt.figure(figsize=(10, 10))
plt.plot(xx, yy, marker='.', color='b', linestyle='none')
plt.quiver(xx, yy, vecs_flat[:, 0], vecs_flat[:, 1], width=0.001)
for seed in seed_points:
    xs, ys = trace_streamline(seed[0], seed[1], vecs, time_step, steps)
    plt.plot(xs, ys, marker='o', linestyle='-', markersize=2)

plt.xlim([0, 19])
plt.ylim([0, 19])
plt.title('Streamlines with Time Step 0.3, Steps 8')
plt.savefig("4/figs/part4_2.png")
plt.show()
