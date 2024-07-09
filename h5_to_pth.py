import h5py
import torch

# Load the H5 file
h5_file_path = 'dqn_model_0.weights.h5'
with h5py.File(h5_file_path, 'r') as h5_file:
    # Assuming the H5 file has datasets named 'data' and 'labels'
    data = h5_file['data'][:]
    labels = h5_file['labels'][:]

# Convert the data to PyTorch tensors
data_tensor = torch.tensor(data)
labels_tensor = torch.tensor(labels)

# Create a dictionary to save the tensors
model_data = {
    'data': data_tensor,
    'labels': labels_tensor
}

# Save the dictionary as a .pth file
pth_file_path = 'dqn_model_0.weights.pth'
torch.save(model_data, pth_file_path)

print(f"Successfully converted {h5_file_path} to {pth_file_path}")
