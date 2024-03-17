import os
import numpy as np

def convert_dat_to_raw(dat_file_path, raw_file_path):
    header_size_bytes = 6  # Size of the header in bytes

    # Open the .dat file in binary read mode
    with open(dat_file_path, 'rb') as dat_file:
        dat_file.seek(header_size_bytes)  # Skip the header

        # Read the rest of the file after the header
        data = np.fromfile(dat_file, dtype=np.uint16)

        # Check if the data is in Little Endian byte order and convert if necessary
        if data.dtype.byteorder == '>':
            data = data.byteswap().newbyteorder()

    # Write the data to a RAW file without the header
    with open(raw_file_path, 'wb') as raw_file:
        data.tofile(raw_file)

# Paths to the files
dat_file_path = '3/data/stagbeetle208x208x123.dat'
raw_file_path = '3/data/stagbeetle208x208x123.raw'

# Convert the DAT file to RAW format with the correct handling
convert_dat_to_raw(dat_file_path, raw_file_path)

print("Conversion completed. The corrected RAW file is saved as:", raw_file_path)

width, height, depth = 208, 208, 123
expected_file_size_bytes = width * height * depth * 2

# Actual file size of the provided RAW file
actual_file_size_bytes = os.path.getsize('3/data/stagbeetle208x208x123.raw')

print(expected_file_size_bytes, actual_file_size_bytes)

# Paths to the files
dat_file_path_2 = '3/data/stagbeetle832x832x494.dat'
raw_file_path_2 = '3/data/stagbeetle832x832x494.raw'

# Call the conversion function
convert_dat_to_raw(dat_file_path_2, raw_file_path_2)

print("Conversion completed. The RAW file is saved as:", raw_file_path_2)

width, height, depth = 832, 832, 494
expected_file_size_bytes = width * height * depth * 2

# Actual file size of the provided RAW file
actual_file_size_bytes = os.path.getsize('3/data/stagbeetle832x832x494.raw')
print(expected_file_size_bytes, actual_file_size_bytes)
