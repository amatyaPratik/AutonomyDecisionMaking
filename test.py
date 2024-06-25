import numpy as np

# Path to the .npy file
file_path = './q_table.npy'

# Load the data from the file
data = np.load(file_path)

# Display the type of data loaded
print(type(data))

# Display the shape of the data if it's an array
if isinstance(data, np.ndarray):
    print('Shape of the array:', data.shape)

# Display the content of the data
print('Data:', data)
