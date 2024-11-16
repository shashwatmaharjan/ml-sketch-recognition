# Import necessary libraries
import os

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data', 'png')
    results_directory = os.path.join(current_directory, 'results', 'binary_classification')

    # List all classes (files) in the data directory
    classes = os.listdir(data_directory)

    # Sort the classes
    classes.sort()

    # Remove the .DS_Store file
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')

    # Print the classes
    print(classes)

    # Choose a class to visualize
    class_name = classes[0]

    # Choose the first file in the class to plot
    filenames = os.listdir(os.path.join(data_directory, class_name))

    # Sort the filenames
    filenames.sort()

    print(filenames)
    

if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()