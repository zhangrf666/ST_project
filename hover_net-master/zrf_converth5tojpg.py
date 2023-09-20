import h5py
from PIL import Image
import os

# Path to the HDF5 file
h5_file_path = '/data/zrf/data/ProcessData_Breast/imgh5/BC23269_C2_cropimage.h5'

# Path to the folder where you want to save the JPG images
output_folder = '/data/zrf/data/ProcessData_Breast/img_jpg_test/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the HDF5 file
with h5py.File(h5_file_path, 'r') as h5_file:
    # Assuming that the image data is stored in a dataset named 'image_data' within the HDF5 file
    image_data = h5_file['images'][:]
    
    for i,img in enumerate(image_data):
        print(img.shape)
    
        # Convert the image data to a PIL Image
        img = Image.fromarray(img)
        
        img.resize((256, 256))
        
        # Save the PIL Image as a JPG file in the output folder
        img.save(os.path.join(output_folder, f'{i}.jpg'))

print('Image saved as output.jpg in', output_folder)
