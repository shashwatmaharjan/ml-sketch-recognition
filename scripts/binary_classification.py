# Import necessary libraries
import os
import json

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    images_data_directory = os.path.join(current_directory, 'data', 'images')
    results_directory = os.path.join(current_directory, 'results', 'binary_classification')

    # Open the .json files with the class names
    with open(os.path.join(data_directory, 'classes.json'), 'r') as file:
        classes = json.load(file)

    # How many results per subclass?
    num_per_subclass = 80

    # Working classes for binary classification
    selected_classes = {}
    selected_classes['household_objects'] = classes['household_objects']
    selected_classes['animals'] = classes['animals']

    # Number of dataset in each class
    num_household_objects = len(selected_classes['household_objects']) * num_per_subclass
    print(f'Number of household objects: {num_household_objects}')

    num_animals = len(selected_classes['animals']) * num_per_subclass
    print(f'Number of animals: {num_animals}')


if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()