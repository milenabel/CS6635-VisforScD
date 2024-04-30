import numpy as np
import matplotlib.pyplot as plt

# Generate the noisy data
np.random.seed(0)  # for reproducibility

# Uniform array from 0 to 1 with 100 points
x = np.linspace(0, 1, 100)

# Function y = sin(10*pi*x) + sin(20*pi*x)
y = np.sin(10 * np.pi * x) + np.sin(20 * np.pi * x)

# Generate 100 instances of noisy y data
noise_std = np.abs(y) / 2  # standard deviation of noise
y_noisy = np.array([y + np.random.normal(0, noise_std, size=y.size) for _ in range(100)])

# Plot each instance as a line
plt.figure(figsize=(10, 6))
for instance in y_noisy:
    plt.plot(x, instance, alpha=0.3)
plt.title('100 Instances of Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('6/figs/100_instances_noisy_data.png')
plt.show()

# Calculate mean and standard deviation
mean_y = np.mean(y_noisy, axis=0)
std_y = np.std(y_noisy, axis=0)

# Plot Mean and one standard deviation
plt.figure(figsize=(10, 6))
plt.plot(x, mean_y, label='Mean of y', color='blue')
plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2, label='Mean ± 1 std dev')
plt.title('Mean and Standard Deviation Over Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('6/figs/mean_std_noisy_data.png')
plt.show()
