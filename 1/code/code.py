import numpy as np
import matplotlib.pyplot as plt

# Part 1.

# Array 1
uniform_array = np.random.uniform(0, 1, 100)

# Array 2
mean = 50  # Adjust this as needed
std_dev = 20  # Adjust this as needed
gaussian_array = np.random.normal(mean, std_dev, 200)

# Create bar chart for uniform data 
plt.boxplot([uniform_array], labels=['Uniform'])
plt.title('Box Plot of Uniform Data')
plt.xlabel('Distribution')
plt.ylabel('Values')
plt.savefig('1/figs/boxplot_U.png')
plt.show()

# Create bar chart for Gaussian data 
plt.boxplot([gaussian_array], labels=['Gaussian'])
plt.title('Box Plot of Gaussian Data')
plt.xlabel('Distribution')
plt.ylabel('Values')
plt.savefig('1/figs/boxplot_G.png')
plt.show()

# Create bar chart for both
plt.boxplot([uniform_array, gaussian_array], labels=['Uniform', 'Gaussian'])
plt.title('Box Plot of Uniform and Gaussian Data')
plt.xlabel('Distribution')
plt.ylabel('Values')
plt.savefig('1/figs/boxplot_both.png')
plt.show()

# Create histogram data
uniform_hist, uniform_bins = np.histogram(uniform_array, bins=20)
gaussian_hist, gaussian_bins = np.histogram(gaussian_array, bins=20)

# Create bar chart for uniform data 
plt.bar(uniform_bins[:-1], uniform_hist, width=0.05, label='Uniform', align='edge')
plt.title('Histogram of Uniform Data')
plt.xlabel('Value Bins')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('1/figs/histogram_U.png')
plt.show()

# Create bar chart for Gaussian data
plt.bar(gaussian_bins[:-1], gaussian_hist, width=3, label='Gaussian', alpha=0.7, align='edge')
plt.title('Histogram of Gaussian Data')
plt.xlabel('Value Bins')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('1/figs/histogram_G.png')
plt.show()

# Create bar chart for both histograms
plt.bar(uniform_bins[:-1], uniform_hist, width=0.05, label='Uniform', align='edge')
plt.bar(gaussian_bins[:-1], gaussian_hist, width=0.05, label='Gaussian', alpha=0.7, align='edge')
plt.title('Histogram of Uniform and Gaussian Data')
plt.xlabel('Value Bins')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('1/figs/histogram_both.png')
plt.show()

# Writing to binary file
with open('data.bin', 'wb') as f:
    np.save(f, uniform_array)
    np.save(f, gaussian_array)

# Reading from binary file
with open('data.bin', 'rb') as f:
    loaded_uniform_array = np.load(f)
    loaded_gaussian_array = np.load(f)
    
# Create cumulative distribution function line graph
sorted_uniform = np.sort(loaded_uniform_array)
sorted_gaussian = np.sort(loaded_gaussian_array)

cdf_uniform = np.arange(1, len(sorted_uniform) + 1) / len(sorted_uniform)
cdf_gaussian = np.arange(1, len(sorted_gaussian) + 1) / len(sorted_gaussian)

# Create CDF for uniform data 
plt.plot(sorted_uniform, cdf_uniform, label='Uniform')
plt.title('Cumulative Distribution Function')
plt.xlabel('Values')
plt.ylabel('CDF')
plt.legend()
plt.savefig('1/figs/CDF_U.png')
plt.show()

# Create CDF for Gaussian data 
plt.plot(sorted_gaussian, cdf_gaussian, label='Gaussian')
plt.title('Cumulative Distribution Function')
plt.xlabel('Values')
plt.ylabel('CDF')
plt.legend()
plt.savefig('1/figs/CDF_G.png')
plt.show()

# Create CDF for both
plt.plot(sorted_uniform, cdf_uniform, label='Uniform')
plt.plot(sorted_gaussian, cdf_gaussian, label='Gaussian')
plt.title('Cumulative Distribution Function')
plt.xlabel('Values')
plt.ylabel('CDF')
plt.legend()
plt.savefig('1/figs/CDF_Both.png')
plt.show()

# Create 2D arrays using uniform random sampling and Gaussian random sampling with 5,000 points on [0,1] x [0,1]
uniform_2d = np.random.rand(5000, 2)
gaussian_2d = np.random.multivariate_normal([0.5, 0.5], [[0.1, 0], [0, 0.1]], 5000)

# Create scatter plot for uniform data 
plt.scatter(uniform_2d[:, 0], uniform_2d[:, 1], label='Uniform 2D')
plt.title('Scatter Plot of 2D Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('1/figs/scplot_U.png')
plt.show()

# Create scatter plot for Gaussian data 
plt.scatter(gaussian_2d[:, 0], gaussian_2d[:, 1], label='Gaussian 2D', alpha=0.7)
plt.title('Scatter Plot of 2D Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('1/figs/scplot_G.png')
plt.show()

# Create scatter plot for both
plt.scatter(uniform_2d[:, 0], uniform_2d[:, 1], label='Uniform 2D')
plt.scatter(gaussian_2d[:, 0], gaussian_2d[:, 1], label='Gaussian 2D', alpha=0.7)
plt.title('Scatter Plot of 2D Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('1/figs/scplot_both.png')
plt.show()

# Generate 100 bins along both dimensions for uniform data
uniform_hist_2d, xedges, yedges = np.histogram2d(uniform_2d[:, 0], uniform_2d[:, 1], bins=100)

# Generate 100 bins along both dimensions for Gaussian data
gaussian_hist_2d, xedges, yedges = np.histogram2d(gaussian_2d[:, 0], gaussian_2d[:, 1], bins=100)

# Show the 2D histograms as images
plt.imshow(uniform_hist_2d.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.title('2D Histogram for Uniform Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig('1/figs/image_U.png')

plt.show()

plt.imshow(gaussian_hist_2d.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.title('2D Histogram for Gaussian Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig('1/figs/image_G.png')
plt.show()

# Show the 2D histograms as contour plots
plt.contourf(xedges[:-1], yedges[:-1], uniform_hist_2d.T, levels=10, cmap='viridis')
plt.title('Contour Plot for Uniform Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig('1/figs/CP_U.png')
plt.show()

plt.contourf(xedges[:-1], yedges[:-1], gaussian_hist_2d.T, levels=10, cmap='viridis')
plt.title('Contour Plot for Gaussian Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig('1/figs/CP_G.png')
plt.show()

# Part 2