def convert_dat_to_raw(dat_file_path, raw_file_path):
    # Open the .dat file in binary read mode and the .raw file in binary write mode
    with open(dat_file_path, 'rb') as dat_file, open(raw_file_path, 'wb') as raw_file:
        # Read the entire content of the .dat file
        data = dat_file.read()
        
        # Iterate through the data 2 bytes at a time (since unsigned short is 2 bytes)
        for i in range(0, len(data), 2):
            # Extract 2 bytes
            value_bytes = data[i:i+2]
            
            # Ensure Little Endian byte order (might be redundant if already in Little Endian)
            value_bytes = value_bytes[::-1]  # Uncomment if byte order conversion is needed
            
            # Write the 2 bytes to the .raw file
            raw_file.write(value_bytes)

# Path to the .dat file you've uploaded
dat_file_path = '3/data/stagbeetle208x208x123.dat'

# Path where the .raw file will be saved
raw_file_path = '3/data/stagbeetle208x208x123.raw'

# Call the conversion function
convert_dat_to_raw(dat_file_path, raw_file_path)

# Path to the .dat file you've uploaded
dat_file_path_2 = '3/data/stagbeetle832x832x494.dat'

# Path where the .raw file will be saved
raw_file_path_2 = '3/data/stagbeetle832x832x494.raw'

# Call the conversion function
convert_dat_to_raw(dat_file_path_2, raw_file_path_2)

print("Conversion completed. The RAW file is saved as:", raw_file_path)
# placeholder for code 