# Import necessary libraries
import os
import json
import numpy as np
import torch

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    npy_files_directory = os.path.join(current_directory, 'data', 'npy')
    results_directory = os.path.join(current_directory, 'results', 'binary_classification')

    # Open the .json files with the class names
    with open(os.path.join(data_directory, 'classes.json'), 'r') as file:
        classes = json.load(file)

    # Working classes for binary classification
    selected_classes = {}
    selected_classes['household_objects'] = classes['household_objects']
    selected_classes['animals'] = classes['animals']

    # Load the .npy files
    household_objects = np.load(os.path.join(npy_files_directory, 'household_objects.npy'))
    animals = np.load(os.path.join(npy_files_directory, 'animals.npy'))

    # Number of dataset in each class
    num_household_objects = household_objects.shape[2]
    print(f'Number of household objects: {num_household_objects}')
    
    num_animals = animals.shape[2]
    print(f'Number of animals: {num_animals}')
     
    # No need to normalize the data since the values are already between 0 and 1
    
    # Reshape the data
    household_objects = household_objects.transpose(2, 0, 1)
    animals = animals.transpose(2, 0, 1)
    
    # Create the labels
    household_objects_labels = np.zeros(num_household_objects)
    animals_labels = np.ones(num_animals)
    
    # Set the device
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        
    else:
        print ("MPS device not found.")
    

if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()