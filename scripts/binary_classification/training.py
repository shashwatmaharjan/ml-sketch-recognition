# Import necessary libraries
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Function to randomly shuffle the data
def shuffle_data(data, labels):
    
    np.random.seed(SEED)
    
    np.random.shuffle(data)
    np.random.shuffle(labels)
    
    return data, labels


# Function to split the data into training, validation, and testing set
def split_data(data, labels):
    
    # Split the data into training, validation, and testing set in ratio 80:10:10
    # Training set
    data_train = data[:, :int(0.8*data.shape[0])]
    labels_train = labels[:int(0.8*data.shape[0])]
    
    # Validation set
    data_val = data[:, int(0.8*data.shape[0]):int(0.9*data.shape[0])]
    labels_val = labels[int(0.8*data.shape[0]):int(0.9*data.shape[0])]
    
    # Testing set
    data_test = data[:, int(0.9*data.shape[0]):]
    labels_test = labels[int(0.9*data.shape[0]):]
    
    return data_train, labels_train, data_val, labels_val, data_test, labels_test

# CNN model
class CNN(nn.Module):
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        # Define convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        
        # Define maxpooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        
        # Define maxpooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define fully connected layer
        # Doubles as a flattening layer
        self.fc1 = nn.Linear(in_features=92*92*32, out_features=1024)
        
        # Define final fully connected layer
        self.fc2 = nn.Linear(in_features=1024, out_features=1)
    
    def forward(self, x):
        
        # Apply the first convolutional layer
        x = F.relu(self.conv1(x))
        
        # Apply the maxpooling layer
        x = self.pool(x)
        
        # Apply the second convolutional layer
        x = F.relu(self.conv2(x))
        
        # Apply the maxpooling layer
        x = self.pool(x)
        
        # Flatten the tensor
        x = x.view(-1, 92*92*64)
        
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        
        # Apply the final fully connected layer with sigmoid activation function
        x = F.sigmoid(self.fc2(x))
        
        return x


# Main function
def main():
    
    global SEED

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    npy_files_directory = os.path.join(current_directory, 'data', 'npy')
    results_directory = os.path.join(current_directory, 'results', 'binary_classification')

    # Open the .json files with the class names
    with open(os.path.join(data_directory, 'classes.json'), 'r') as file:
        classes = json.load(file)

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
    
    # Print the shape of the data
    print(f'Household objects shape: {household_objects.shape}')
    print(f'Animals shape: {animals.shape}')
    
    # Create the labels
    household_objects_labels = np.zeros(num_household_objects)
    animals_labels = np.ones(num_animals)
    
    # Stack the data
    data = np.vstack((household_objects, animals))
    labels = np.hstack((household_objects_labels, animals_labels))
    
    # Randomize the data with a SEED
    SEED = 42
    data, labels = shuffle_data(data, labels)
    
    # Split the data into training, validation, and testing set in ratio 80:10:10
    data_train, labels_train, data_val, labels_val, data_test, labels_test = split_data(data, labels)
    
    # Set the device
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        
    else:
        print ("MPS device not found.")
        
    # Instantiate the model
    model = CNN().to(mps_device)
    

if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()