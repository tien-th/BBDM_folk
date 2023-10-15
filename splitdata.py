import os
import numpy as np 
import shutil

src_dir = '/workdir/carrot/data_120k_1'
dst_dir = ''

# Define the paths for the train, val, and test directories
train_dir = dst_dir + '/train'
val_dir = dst_dir + '/val'
test_dir = dst_dir + '/test'

# Create subdirectories for CT and PET images in train, val, and test
for dataset_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(dataset_dir, 'A'), exist_ok=True) # CT
    os.makedirs(os.path.join(dataset_dir, 'B'), exist_ok=True) # PET

# Split the dataset
train_size = 96000
val_size = 12000
test_size = 12000

# Train set
for i in range(train_size):
    ct_filename = os.path.join(train_dir, 'A', f"{i}.npy") # CT
    pet_filename = os.path.join(train_dir, 'B', f"{i}.npy") # PET
    # copy ct data from src to dst
    shutil.copy(os.path.join(src_dir, 'CT', f"{i}.npy"), ct_filename)
    # copy pet data from src to dst
    shutil.copy(os.path.join(src_dir, 'PET', f"{i}.npy"), pet_filename)
    

# Validation set
for i in range(train_size, train_size + val_size):
    ct_filename = os.path.join(val_dir, 'A', f"{i}.npy")
    pet_filename = os.path.join(val_dir, 'B', f"{i}.npy")
    # copy ct data from src to dst
    shutil.copy(os.path.join(src_dir, 'CT', f"{i}.npy"), ct_filename)
    # copy pet data from src to dst
    shutil.copy(os.path.join(src_dir, 'PET', f"{i}.npy"), pet_filename)

# Test set
for i in range(train_size + val_size, train_size + val_size + test_size):
    ct_filename = os.path.join(test_dir, 'A', f"{i}.npy")
    pet_filename = os.path.join(test_dir, 'B', f"{i}.npy")
    # copy ct data from src to dst
    shutil.copy(os.path.join(src_dir, 'CT', f"{i}.npy"), ct_filename)
    # copy pet data from src to dst
    shutil.copy(os.path.join(src_dir, 'PET', f"{i}.npy"), pet_filename)
