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
linear_interpolation = False # else, midpoint method

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
    8: [(1, 0.5, 0.5, 0)], 9: [(1, 0.5, 0, 0.5)], 10: [(1, 0.5, 0.5, 0), (0.5, 1, 0, 0.5)], 11: [(0, 0.5, 0.5, 0)],
    12: [(0.5, 1, 0.5, 0)], 13: [(0, 0.5, 0.5, 1)], 14: [(0.5, 1, 1, 0.5)], 15: []
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

# Draw Lines in Marching Squares - Linear Interpolation
def march_sq_lin_interp(data, i, j, isovalue):
    # TODO Part 3
    return

# # Save the plot to a file
# output_file4c = "2/figs/plot_4c.png"
# plt.savefig(output_file4c)
# output_file4c

# Implement simple marching squares with midpoint approach
for i in x[0:-1]:
    for j in y[0:-1]:
        if (linear_interpolation):
            march_sq_lin_interp(data, i, j, isovalue)
        else:
            march_sq_midpoint(data, i, j, isovalue)


plt.show()

# Save the plot to a file
output_file4b = "2/figs/plot_4b.png"
plt.savefig(output_file4b)
output_file4b