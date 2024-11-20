# Import necessary libraries
import os
import json
import matplotlib.pyplot as plt

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    images_data_directory = os.path.join(current_directory, 'data', 'images')

    # Open the .json files with the class names
    with open(os.path.join(data_directory, 'classes.json'), 'r') as file:
        classes = json.load(file)
    
    # How many results per subclass?
    num_per_subclass = 80

    # Create empty dictionary with keys from the classes
    classes_statistics = {key: {} for key in classes.keys()}

    # Go through each of the classes and see how many subclasses are in each class
    for major_class in classes.keys():

        classes_statistics[major_class] = len(classes[major_class])*num_per_subclass
    
    # Plot histogram
    major_classes = list(classes_statistics.keys())
    counts = list(classes_statistics.values())

    plt.figure(figsize=(10,6))
    plt.bar(major_classes, counts, color='blue')
    plt.xlabel('Major Classes')
    plt.ylabel('Number of Subclasses')
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(data_directory, 'subclass_count.png'), dpi=300)

    plt.show()


if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Set default figure size for plots
    plt.rcParams['figure.figsize'] = (8, 6)

    # Set default font and math text font for the plots
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Define font size for plots
    plt.rcParams.update({'font.size': 15})

    # Default grid opacity
    plt.rcParams['grid.alpha'] = 0.2

    # Call the main function
    main()