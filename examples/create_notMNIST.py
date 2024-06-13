import os
import numpy as np
import torch
import deeplake
import subprocess
import argparse

# Courtesy: David Flanagan
# https://github.com/davidflanagan/notMNIST-to-MNIST/blob/master/convert_to_mnist_format.py

# Function to write label data in ubyte format
def write_labeldata(labeldata, outputfile):
    header = np.array([0x0801, len(labeldata)], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())

# Function to write image data in ubyte format
def write_imagedata(imagedata, outputfile):
    header = np.array([0x0803, len(imagedata), 28, 28], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())

def main(data_dir):
    # Create the directory for the notMNIST dataset if it does not exist
    notmnist_dir = os.path.join(data_dir, 'notMNIST', 'raw')
    if not os.path.exists(notmnist_dir):
        os.makedirs(notmnist_dir)
    
    # Load notMNIST dataset from deeplake
    ds = deeplake.load('hub://activeloop/not-mnist-small')

    # Manually create the idx_to_class mapping for notMNIST
    idx_to_class = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}
    
    # Add the idx_to_class mapping to the dataset object
    ds.class_to_idx = idx_to_class

    # Specify the decode method
    decode_method = {'images': 'numpy'}

    # Get the dataloader with the specified decode method
    dataloader = ds.pytorch(num_workers=16, batch_size=len(ds), shuffle=True, decode_method=decode_method)

    # Fetch the images and targets from the dataloader
    batch = next(iter(dataloader))
    images, targets = batch['images'], batch['labels']

    # Convert images and targets to numpy arrays
    images_np = images.numpy()
    targets_np = targets.numpy().astype(np.uint8).squeeze()

    # Define the paths for the ubyte files
    image_file_path = os.path.join(notmnist_dir, 't10k-images-idx3-ubyte')
    label_file_path = os.path.join(notmnist_dir, 't10k-labels-idx1-ubyte')

    # Save the images and labels in the classical MNIST format
    write_imagedata(images_np, image_file_path)
    write_labeldata(targets_np, label_file_path)

    # Compress all -ubyte files in the folder using gzip without overwriting the original files
    for file in os.listdir(notmnist_dir):
        if file.endswith("-ubyte"):
            file_path = os.path.join(notmnist_dir, file)
            compressed_file_path = f"{file_path}.gz"
            print(f"Compressing {file_path} to {compressed_file_path}")
            subprocess.run(f"gzip -c {file_path} > {compressed_file_path}", shell=True, check=True)

    print("Data processing and compression completed successfully.")
    print(f"class_to_idx: {ds.class_to_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create, process and compress notMNIST dataset.')
    parser.add_argument('data_dir', type=str, help='Directory to store the notMNIST dataset.')
    args = parser.parse_args()
    main(args.data_dir)

# Run the script to create the notMNIST dataset like this
# python create_notMNIST.py ./data
