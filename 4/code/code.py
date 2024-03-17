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

# Define function to plot streamlines with varying time steps and steps
def plot_streamlines(seed_points, vecs, time_step, steps, title, file_name):
    plt.figure(figsize=(10, 10))
    plt.plot(xx, yy, marker='.', color='b', linestyle='none')
    plt.quiver(xx, yy, vecs_flat[:, 0], vecs_flat[:, 1], width=0.001)
    for seed in seed_points:
        xs, ys = trace_streamline(seed[0], seed[1], vecs, time_step, steps)
        plt.plot(xs, ys, marker='o', linestyle='-', markersize=2)

    plt.xlim([0, 19])
    plt.ylim([0, 19])
    plt.title(title)

    # Save the plot
    plot_path = f"4/figs/{file_name}.png"
    plt.savefig(plot_path)
    return plot_path

# Parameters for different streamlines
streamline_params = [
    (0.15, 16, 'Streamlines with Time Step 0.15, Steps 16', 'part4_streamlines_step0.15_steps16'),
    (0.075, 32, 'Streamlines with Time Step 0.075, Steps 32', 'part4_streamlines_step0.075_steps32'),
    (0.0375, 64, 'Streamlines with Time Step 0.0375, Steps 64', 'part4_streamlines_step0.0375_steps64')
]

# Generate and save plots for each parameter set
plot_paths = [plot_streamlines(seed_points, vecs, ts, st, title, file_name) for ts, st, title, file_name in streamline_params]
plt.show()
plot_paths



#extra credit

# Parameters for different streamlines
streamline_params_2 = [
    (0.3, 8, 'Streamlines with Time Step 0.3, Steps 8', 'part4_streamlines_step0.3_steps8'),
    (0.15, 16, 'Streamlines with Time Step 0.15, Steps 16', 'part4_streamlines_step0.15_steps16'),
    (0.075, 32, 'Streamlines with Time Step 0.075, Steps 32', 'part4_streamlines_step0.075_steps32'),
    (0.0375, 64, 'Streamlines with Time Step 0.0375, Steps 64', 'part4_streamlines_step0.0375_steps64')
]

def runge_kutta_integration(x, y, vecs, time_step):
    """
    Implement the 4th order Runge-Kutta (RK4) method for integrating the vector field.
    Args:
    - x, y: Initial position
    - vecs: Vector field data
    - time_step: Time step value
    
    Returns:
    - New x, y positions after the time step
    """
    # Helper to safely get vector considering boundaries
    def get_vector(x, y, vecs):
        if 0 <= x < 20 and 0 <= y < 20:
            ix, iy = int(x), int(y)
            return bilinear_interpolation(x, y, vecs)
        return np.array([0, 0])

    # k1
    v1 = get_vector(x, y, vecs)
    k1x, k1y = v1

    # k2
    v2 = get_vector(x + k1x * time_step / 2, y + k1y * time_step / 2, vecs)
    k2x, k2y = v2

    # k3
    v3 = get_vector(x + k2x * time_step / 2, y + k2y * time_step / 2, vecs)
    k3x, k3y = v3

    # k4
    v4 = get_vector(x + k3x * time_step, y + k3y * time_step, vecs)
    k4x, k4y = v4

    # Combine
    x_new = x + (k1x + 2*k2x + 2*k3x + k4x) * time_step / 6
    y_new = y + (k1y + 2*k2y + 2*k3y + k4y) * time_step / 6

    return x_new, y_new

def trace_streamline_rk4(x, y, vecs, time_step, steps):
    """
    Trace a streamline from a seed point using the Runge-Kutta method.
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
        x, y = runge_kutta_integration(x, y, vecs, time_step)
        # Boundary conditions
        if x < 0 or x >= 20 or y < 0 or y >= 20:
            break
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Generate and save plots for each parameter set using RK4
rk4_plot_paths = []
for ts, st, title, file_name in streamline_params_2:
    plt.figure(figsize=(10, 10))
    plt.plot(xx, yy, marker='.', color='b', linestyle='none')
    plt.quiver(xx, yy, vecs_flat[:, 0], vecs_flat[:, 1], width=0.001)
    for seed in seed_points:
        xs, ys = trace_streamline_rk4(seed[0], seed[1], vecs, ts, st)
        plt.plot(xs, ys, marker='o', linestyle='-', markersize=2)

    plt.xlim([0, 19])
    plt.ylim([0, 19])
    plt.title(title + ' (RK4)')

    # Save the plot
    rk4_plot_path = f"4/figs/{file_name}_rk4.png"
    plt.savefig(rk4_plot_path)
    rk4_plot_paths.append(rk4_plot_path)
    plt.show()

rk4_plot_paths


# 

