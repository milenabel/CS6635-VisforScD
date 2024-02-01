import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
import pandas as pd
from pandas.plotting import parallel_coordinates

# Part 1: Creating Arrays
uniform_array = np.random.uniform(0, 1, 100)
gaussian_array = np.random.normal(50, 20, 200)  # Mean = 50, Std Dev = 20

# Part 1: Box Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot(uniform_array)
plt.title('Box Plot of Uniform Data')
plt.subplot(1, 2, 2)
plt.boxplot(gaussian_array)
plt.title('Box Plot of Gaussian Data')
plt.savefig('1/figs/boxplots.png')
plt.show()

# Part 2: Histograms without Histogram Function
# Manually calculating histogram data
def manual_histogram(data, bins):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = (max_val - min_val) / bins
    hist = np.zeros(bins)
    for d in data:
        index = int((d - min_val) / range_val)
        index = min(index, bins - 1)  # Ensure the index is within the range
        hist[index] += 1
    return hist, np.linspace(min_val, max_val, bins+1)

# Calculate histograms
uniform_hist, uniform_bins = manual_histogram(uniform_array, 20)
gaussian_hist, gaussian_bins = manual_histogram(gaussian_array, 20)

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(uniform_bins[:-1], uniform_hist, width=np.diff(uniform_bins), edgecolor='black', align='edge')
plt.title('Histogram of Uniform Data')
plt.subplot(1, 2, 2)
plt.bar(gaussian_bins[:-1], gaussian_hist, width=np.diff(gaussian_bins), edgecolor='black', align='edge')
plt.title('Histogram of Gaussian Data')
plt.savefig('1/figs/histograms.png')
plt.show()

# Part 3: Binary File I/O
with open('data.bin', 'wb') as f:
    np.save(f, uniform_array)
    np.save(f, gaussian_array)

with open('data.bin', 'rb') as f:
    loaded_uniform_array = np.load(f)
    loaded_gaussian_array = np.load(f)

# Part 3: Cumulative Distribution Function
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.sort(loaded_uniform_array), np.linspace(0, 1, len(loaded_uniform_array)))
plt.title('CDF of Uniform Data')
plt.subplot(1, 2, 2)
plt.plot(np.sort(loaded_gaussian_array), np.linspace(0, 1, len(loaded_gaussian_array)))
plt.title('CDF of Gaussian Data')
plt.savefig('1/figs/CDFs.png')
plt.show()

# Part 4: 2D Arrays and Scatter Plots
uniform_2d = np.random.rand(5000, 2)
gaussian_2d = np.random.multivariate_normal([0.5, 0.5], [[0.1, 0], [0, 0.1]], 5000)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(uniform_2d[:, 0], uniform_2d[:, 1], alpha=0.6)
plt.title('Uniform 2D Scatter Plot')
plt.subplot(1, 2, 2)
plt.scatter(gaussian_2d[:, 0], gaussian_2d[:, 1], alpha=0.6)
plt.title('Gaussian 2D Scatter Plot')
plt.savefig('1/figs/2Dscatterplots.png')
plt.show()

# Part 4b: 2D Histograms as Images
def histogram2d_to_image(data, bins):
    hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=bins)
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
    plt.colorbar()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
histogram2d_to_image(uniform_2d, 100)
plt.title('2D Histogram for Uniform Data')
plt.subplot(1, 2, 2)
histogram2d_to_image(gaussian_2d, 100)
plt.title('2D Histogram for Gaussian Data')
plt.savefig('1/figs/2Dhistograms.png')
plt.show()

# Part 4c: Contour Plots
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.tricontourf(uniform_2d[:, 0], uniform_2d[:, 1], np.zeros(len(uniform_2d)), levels=10, cmap='viridis')
# plt.title('Uniform 2D Contour Plot')
# plt.savefig('../figs/2Dcontourplots.png')
# plt.subplot

# Part 4c: Contour Plots for 2D Arrays without saving

plt.figure(figsize=(12, 6))

# Contour plot for uniform data
plt.subplot(1, 2, 1)
plt.tricontourf(uniform_2d[:, 0], uniform_2d[:, 1], np.zeros(len(uniform_2d)), levels=10, cmap='viridis')
plt.title('Uniform 2D Contour Plot')
plt.colorbar()

# Contour plot for Gaussian data
plt.subplot(1, 2, 2)
plt.tricontourf(gaussian_2d[:, 0], gaussian_2d[:, 1], np.zeros(len(gaussian_2d)), levels=10, cmap='viridis')
plt.title('Gaussian 2D Contour Plot')
plt.colorbar()
plt.savefig('1/figs/2Dcontourplots.png')
plt.show()

# Part 2

# Read the data from the CSV file, skipping the first 4 rows
data = pd.read_csv('1/data/NOAA-Temperatures.csv', skiprows=4)

# Extract the Year and Value columns
years = data['Year']
temperatures = data['Value']

# Create a list of colors based on positive/negative temperature changes
colors = ['blue' if temp < 0 else 'red' for temp in temperatures]

# Create the bar plot
plt.figure(figsize=(12, 6))
plt.bar(years, temperatures, color=colors, width=0.5)
plt.xlabel('Year')
plt.ylabel('Degrees F +/- 1 From Average')
plt.title('NOAA Land Ocean Temperature Anomalies (1880-2017)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('1/figs/NOAA.png')
plt.show()

# Read the dataset
cereal_df = pd.read_excel('1/data/Breakfast-Cereals.xls')

# Selecting the three cereals
selected_cereals = selected_cereals = ['Special K', 'Puffed Rice', 'Cheerios']

# Extracting the data for the selected cereals
selected_data = cereal_df[cereal_df['Cereal'].isin(selected_cereals)]

# Nutritional attributes to compare
attributes = ['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber', 'Carbohydrates', 'Sugars', 'Potassium']

# Extracting the values for the attributes
values = selected_data[attributes].values

# Number of variables we're plotting
num_vars = len(attributes)

# Split the circle into even parts and save angles so we know where to put each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
values = np.concatenate((values, values[:,[0]]), axis=1)
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], attributes)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100, 120], ["20", "40", "60", "80", "100", "120"], color="grey", size=7)
plt.ylim(0,140)

# Plot each cereal's line
for i in range(len(selected_cereals)):
    ax.plot(angles, values[i], linewidth=1, linestyle='solid', label=selected_cereals[i])
    ax.fill(angles, values[i], alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('1/figs/Cereals.png')
plt.show()

# Load the uploaded datasets
births_df = pd.read_csv('1/data/US_births_2000-2014_SSA.csv')
covid_geo_df = pd.read_csv('1/data/mmsa-icu-beds.csv')

# Displaying the first few rows of each dataset for understanding their structure
# births_df.head(), covid_geo_df.head()

# Scatter Plot for US Births data
plt.figure(figsize=(12, 6))
sns.scatterplot(data=births_df, x='date_of_month', y='births', hue='day_of_week', palette='viridis')
plt.title('US Births by Day of Month and Day of Week (2000-2014)')
plt.xlabel('Date of Month')
plt.ylabel('Number of Births')
plt.legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('1/figs/USbirths.png')
plt.show()

# Cleaning covid_geo_df data (removing % sign and converting to float)
covid_geo_df['total_percent_at_risk'] = covid_geo_df['total_percent_at_risk'].str.rstrip('%').astype('float')

# Group the 'total_percent_at_risk' into bins of 10% intervals for better visualization
covid_geo_df['risk_group'] = pd.cut(covid_geo_df['total_percent_at_risk'], 
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                                             '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                                     include_lowest=True)

# Recreate the Parallel Coordinates Plot for COVID-19 High-Risk Geography data
plt.figure(figsize=(12, 6))
parallel_coordinates(covid_geo_df.dropna(), class_column='risk_group', 
                     cols=['high_risk_per_ICU_bed', 'high_risk_per_hospital', 'icu_beds', 'hospitals'], 
                     colormap='viridis', alpha=0.5)
plt.title('COVID-19 High-Risk Geography Data')
plt.ylabel('Values')
plt.xlabel('Metrics')
plt.legend(title='Total Percent at Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('1/figs/COVID19.png')
plt.show()


# Part 4
# Load the MRI data
mri_path = '1/data/T2.nii.gz'
mri_data = nib.load(mri_path)

# Get the data as a numpy array
mri_array = mri_data.get_fdata()

# Extract one slice for each axis
slice_x = mri_array[160, :, :]  # Middle slice on the x-axis
slice_y = mri_array[:, 160, :]  # Middle slice on the y-axis
slice_z = mri_array[:, :, 128]  # Middle slice on the z-axis

# Define two colormaps
colormaps = ['gray', 'viridis']

# Plotting the slices with the two colormaps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, cmap in enumerate(colormaps):
    axes[i, 0].imshow(slice_x.T, cmap=cmap, origin='lower')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Slice X - {cmap.capitalize()} colormap')

    axes[i, 1].imshow(slice_y.T, cmap=cmap, origin='lower')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'Slice Y - {cmap.capitalize()} colormap')

    axes[i, 2].imshow(slice_z.T, cmap=cmap, origin='lower')
    axes[i, 2].axis('off')
    axes[i, 2].set_title(f'Slice Z - {cmap.capitalize()} colormap')

plt.tight_layout()
plt.savefig('1/figs/MRI.png')
plt.show()
