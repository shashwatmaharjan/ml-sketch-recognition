# Import necessary libraries
import os
import json
import numpy as np
import matplotlib.image as mpimg

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    images_data_directory = os.path.join(current_directory, 'data', 'images')
    npy_files_directory = os.path.join(current_directory, 'data', 'npy')
    
    # Create the directory if it does not exist
    if not os.path.exists(npy_files_directory):
        os.makedirs(npy_files_directory)

    # Open the .json files with the class names
    with open(os.path.join(data_directory, 'classes.json'), 'r') as file:
        classes = json.load(file)
    
    # Loop through all the classes and convert the images to .npy files
    for class_name in classes.keys():
        
        # Create the directory if it does not exist
        if not os.path.exists(os.path.join(npy_files_directory, class_name)):
            os.makedirs(os.path.join(npy_files_directory, class_name))
        
        # Loop through all the subclasses
        for subclass in classes[class_name]:
            
            # List all the images in the directory with .png extension
            all_images = os.listdir(os.path.join(images_data_directory, subclass))
            
            # Sort the images
            all_images.sort()
            
            # Loop through the images to then convert them to .npy files
            for n_image, image in enumerate(all_images):
                
                if n_image == 0:
                
                    # Load the image
                    img = mpimg.imread(os.path.join(images_data_directory, subclass, image))
                    
                    # Convert the image to a numpy array
                    img_array = np.array(img)
                
                else:
                    
                    # Load the image
                    img = mpimg.imread(os.path.join(images_data_directory, subclass, image))
                    
                    # Convert the image to a numpy array
                    img_array = np.dstack((img_array, img))
            
            # Print a message
            print(f'Class: {class_name} - Subclass: {subclass} Finished!')
            
            # Save the numpy array
            np.save(os.path.join(npy_files_directory, class_name, f'{subclass}.npy'), img_array)
    
    # Print a message
    print('All images have been converted to .npy files!')


if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()