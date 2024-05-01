import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

# Set a random seed for reproducibility
np.random.seed(0)

# Generate x values
x = np.linspace(0, 1, 100)

# Compute y values
y = np.sin(10 * np.pi * x) + np.sin(20 * np.pi * x)
y_exact = np.sin(10 * np.pi * x)  # Exact function

# Generate noisy data
noise_std = np.abs(y) / 2
y_noisy = np.array([y + np.random.normal(0, noise_std, size=y.size) for _ in range(100)])

# Calculate mean and standard deviation of the noisy data
mean_y = np.mean(y_noisy, axis=0)
std_y = np.std(y_noisy, axis=0)

# Plot the noisy data instances
plt.figure(figsize=(10, 6))
for instance in y_noisy:
    plt.plot(x, instance, alpha=0.3)
plt.title('100 Instances of Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('6/figs/100_instances_noisy_data.png')
plt.show()

# Plot the mean and standard deviation
plt.figure(figsize=(10, 6))
plt.plot(x, mean_y, color='blue', label='Mean of y')
plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='blue', alpha=0.2, label='Mean ± 1 std dev')
plt.title('Mean and Standard Deviation Over Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('6/figs/mean_std_noisy_data.png')
plt.show()

# FFT analysis
y_fft = scipy.fftpack.fft(mean_y)
frequencies = scipy.fftpack.fftfreq(len(mean_y), x[1] - x[0])

# Plot frequency spectrum
plt.figure(figsize=(10, 6))
plt.stem(frequencies, np.abs(y_fft), linefmt='b-', markerfmt="bo", basefmt="-b")
plt.title('Frequency Spectrum of the Mean Data')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.xlim(-10, 10)
plt.savefig('6/figs/frequency_spectrum_mean_data.png')
plt.show()

# Apply a Butterworth bandpass filter
b, a = scipy.signal.butter(N=2, Wn=[4, 6], btype='band', fs=100)
filtered_y = scipy.signal.filtfilt(b, a, mean_y)

# Plot the exact function and filtered data
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, label='Exact Data (sin(10*pi*x))', color='green')
plt.plot(x, filtered_y, label='Filtered Data', color='red')
plt.title('Comparison of Exact Data and Filtered Signal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('6/figs/exact_data_filtered_signal.png')
plt.show()
