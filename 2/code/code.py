#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Draws the starting plot. Don't change this code
def draw_initial_plot(data, x, y):

    # Draw grid and hide labels
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.set_xlim(-.5, len(x)-.5)
    ax.set_ylim(-.5, len(y)-.5)
    ax.grid(True)
    plt.xticks(np.arange(-.5,data.shape[0], step=1))
    plt.yticks(np.arange(-.5,data.shape[1], step=1))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # Add text on cells
    for i in range(len(x)):
        for j in range(len(y)):
            ax.text(i,j+.1, str(int(data[i,j])), ha='center', va='bottom', size=18)

    return fig, ax


# Load data and make range arrays for looping
data = np.load("2/data/scalars_2D.npy") # access scalar values by data[i,j]
x = np.arange(0,data.shape[0])
y = np.arange(0,data.shape[1])

fig, ax = draw_initial_plot(data, x, y)


# Draws a dot at a given point
# point - type of tuple or array with length 2: (x,y) or [x,y]
# color - string of color to draw dot - Ex: ""red", "green", "blue"
def draw_dot(point, color):
    ax.scatter(point[0], point[1], color=color)

# Draws a line from point0 to point1
# point0 - type of tuple or array with length 2: (x,y) or [x,y]
# point1 - type of tuple or array with length 2: (x,y) or [x,y]
def draw_line(point0, point1):
    x = [point0[0], point1[0]]
    y = [point0[1], point1[1]]
    ax.plot(x, y, color="black")



#-----------------------
# ASSIGNMENT STARTS HERE
#-----------------------
isovalue = 50
linear_interpolation = True # else, midpoint method

# Add colored points to identify if cells are below or above the isovalue threshold
for i in x:
    for j in y:
        cell_value = data[i, j]
        color = "red" if cell_value > isovalue else "black"
        draw_dot((i, j), color)

# Save the plot to a file
# output_file4a = "2/figs/plot_4a.png"
# plt.savefig(output_file4a)
# output_file4a
        

# Marching Squares Lookup Table
# Each number represents a configuration of the 4 corners of a cell
# (top-left, top-right, bottom-right, bottom-left) in binary. 1 means above the isovalue, 0 means below.        
lookup_table = {
    0: [], 1: [(0.5, 0, 0, 0.5)], 2: [(0, 0.5, 0.5, 1)], 3: [(0.5, 0, 0.5, 1)],
    4: [(0.5, 1, 1, 0.5)], 5: [(0.5, 0, 1, 0.5), (0, 0.5, 0.5, 1)], 6: [(0, 0.5, 1, 0.5)], 7: [(0.5, 0, 1, 0.5)],
    8: [(1, 0.5, 0.5, 0)], 9: [(1, 0.5, 0, 0.5)], 10: [(1, 0.5, 0.5, 0), (0.5, 1, 0, 0.5)], 11: [(0.5, 1, 1, 0.5)],
    12: [(0.5, 1, 0.5, 0)], 13: [(0, 0.5, 0.5, 1)], 14: [(0, 0.5, 0.5, 0)], 15: []
}


# Draw Lines in Marching Squares - Midpoint
def march_sq_midpoint(data, i, j, isovalue):
    # return
    index = 0
    if data[i, j] > isovalue:
        index |= 1
    if data[i, j+1] > isovalue:
        index |= 2
    if data[i+1, j+1] > isovalue:
        index |= 4
    if data[i+1, j] > isovalue:
        index |= 8
    for line in lookup_table[index]:
        point0 = (line[0] + i, line[1] + j)
        point1 = (line[2] + i, line[3] + j)
        draw_line(point0, point1)

# Helper function to interpolate between two points
# def interpolate(p1, p2, v1, v2, isovalue):
#     if v1 == v2:
#         return p1
#     return (p1[0] + (p2[0] - p1[0]) * (isovalue - v1) / (v2 - v1),
#             p1[1] + (p2[1] - p1[1]) * (isovalue - v1) / (v2 - v1))

# Draw Lines in Marching Squares - Linear Interpolation
# def march_sq_lin_interp(data, i, j, isovalue):
#     # Function to interpolate the point along a grid line
#     def interp_point(p1, p2, v1, v2):
#         if v2 - v1 == 0:
#             return p1  # Avoid division by zero
#         t = (isovalue - v1) / (v2 - v1)
#         return p1 + t * (p2 - p1)

#     # Determine the index of the cell
#     index = 0
#     if data[i, j] > isovalue:
#         index |= 1
#     if data[i, j+1] > isovalue:
#         index |= 2
#     if data[i+1, j+1] > isovalue:
#         index |= 4
#     if data[i+1, j] > isovalue:
#         index |= 8

#     # Get the line segments for this cell configuration
#     lines = lookup_table[index]

#     for line in lines:
#         # Corners of the cell
#         corners = [(i, j), (i, j+1), (i+1, j+1), (i+1, j)]

#         # Scalar values at the corners
#         values = [data[i, j], data[i, j+1], data[i+1, j+1], data[i+1, j]]

#         # Interpolate points
#         p0 = interp_point(np.array(corners[line[0]]), np.array(corners[line[1]]), 
#                           values[line[0]], values[line[1]])
#         p1 = interp_point(np.array(corners[line[2]]), np.array(corners[line[3]]), 
#                           values[line[2]], values[line[3]])

#         draw_line(p0, p1)
        

# Edge Lookup Table
# edge_table = {
#     0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(1, 3)],
#     4: [(1, 2)], 5: [(0, 1), (2, 3)], 6: [(0, 2)], 7: [(2, 3)],
#     8: [(2, 3)], 9: [(0, 2)], 10: [(0, 3), (1, 2)], 11: [(1, 2)],
#     12: [(1, 3)], 13: [(0, 1)], 14: [(0, 3)], 15: []
# }
        
# edge_table = {
#     0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(3, 0), (0, 1)],
#     4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 1), (1, 2)], 7: [(3, 2)],
#     8: [(2, 3)], 9: [(2, 3), (0, 1)], 10: [(2, 3), (3, 0)], 11: [(2, 1)],
#     12: [(3, 0), (1, 2)], 13: [(1, 2)], 14: [(3, 2)], 15: []
# }
        
# # Linear Interpolation function
# def lerp(a, b, v):
#     return a + (b - a) * v
        
# # Draw Lines in Marching Squares - Linear Interpolation
# def march_sq_lin_interp(data, i, j, isovalue):
#     index = 0
#     if data[i, j] > isovalue:
#         index |= 1
#     if data[i, j+1] > isovalue:
#         index |= 2
#     if data[i+1, j+1] > isovalue:
#         index |= 4
#     if data[i+1, j] > isovalue:
#         index |= 8

#     # Map edge tuples to indices
#     edge_to_point = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}

#     # Check each edge for intersection
#     def edge_point(p1, p2):
#         val1 = data[i + p1[0], j + p1[1]]
#         val2 = data[i + p2[0], j + p2[1]]
#         if (val1 > isovalue) != (val2 > isovalue):
#             t = (isovalue - val1) / (val2 - val1)
#             return (lerp(p1[0], p2[0], t) + i, lerp(p1[1], p2[1], t) + j)
#         return None

#     # Top, Right, Bottom, Left
#     edges = [(0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0), (1, 0, 0, 0)]
#     points = [edge_point(edges[k][0:2], edges[k][2:4]) for k in range(4)]

#     # Draw lines based on the lookup table
#     for line in lookup_table[index]:
#         p0, p1 = edge_to_point[(line[0], line[1])], edge_to_point[(line[2], line[3])]
#         if points[p0] and points[p1]:
#             draw_line(points[p0], points[p1])


# Modify the march_sq_lin_interp function
        


# Linear interpolation function
def interp_point(p1, p2, v1, v2, isovalue):
    if v1 == v2:  # To avoid division by zero
        return p1
    t = (isovalue - v1) / (v2 - v1)
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

# Marching squares algorithm with linear interpolation
def march_sq_lin_interp(data, i, j, isovalue, edge_lookup_table):

    # Lookup table for the edges to interpolate
    edge_lookup_table = {
        0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)],
        4: [(1, 2)], 5: [(3, 2), (0, 1)], 6: [(1, 2)], 7: [(3, 2)],
        8: [(2, 3)], 9: [(2, 3)], 10: [(0, 1), (2, 3)], 11: [(2, 1)],
        12: [(1, 3)], 13: [(1, 3)], 14: [(0, 3)], 15: []
    }
    # Determine the index of the cell
    index = 0
    if data[i, j] < isovalue: index |= 1
    if data[i, j+1] < isovalue: index |= 2
    if data[i+1, j+1] < isovalue: index |= 4
    if data[i+1, j] < isovalue: index |= 8

    # Get the edges where the contour intersects
    edges = edge_lookup_table[15 - index]  # Invert index because we want the complement
    edge_points = []

    # Define the grid points (corners)
    corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
    # Define the values at the corners
    values = [data[corner] for corner in corners]

    # Define the edges by pairs of corners
    edge_to_corners = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Calculate the intersection points
    for edge in edges:
        corner1, corner2 = edge_to_corners[edge[0]], edge_to_corners[edge[1]]
        p1, p2 = corners[corner1[0]], corners[corner2[0]]
        v1, v2 = values[corner1[0]], values[corner2[0]]
        edge_points.append(interp_point(p1, p2, v1, v2, isovalue))

    # Draw the lines between the intersection points
    if len(edge_points) == 2:
        draw_line(edge_points[0], edge_points[1])



# # Implement simple marching squares with midpoint approach
for i in x[0:-1]:
    for j in y[0:-1]:
        if (linear_interpolation):
            march_sq_lin_interp(data, i, j, isovalue)
        else:
            march_sq_midpoint(data, i, j, isovalue)

# # Save the plot to a file
# output_file4b = "2/figs/plot_4b.png"
# plt.savefig(output_file4b)
# output_file4b
            
# Save the plot to a file
output_file4c = "2/figs/plot_4c.png"
plt.savefig(output_file4c)
output_file4c

plt.show()