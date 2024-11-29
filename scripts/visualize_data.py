# Import necessary libraries
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to plot the png images
def plot_images(filename, class_name, data_directory):

    # Load the image
    img = mpimg.imread(os.path.join(data_directory, class_name, filename))

    # Plot the image on a white background
    imgplot = plt.imshow(img, cmap='gray', aspect='auto')

    # Remove the axis
    plt.axis('off')

    # Set the title
    plt.title(class_name)

    # Remove the axis
    plt.show()

    # Return the dimensions of the image
    return img.shape


# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data', 'images')

    # List all classes (files) in the data directory
    classes = os.listdir(data_directory)

    # Sort the classes
    classes.sort()
    
    # Remove the .DS_Store file
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')

    # Choose a class to visualize
    class_name = classes[0]

    # Choose the first file in the class to plot
    filenames = os.listdir(os.path.join(data_directory, class_name))

    # Sort the filenames
    filenames.sort()
    
    # Remove the .DS_Store file
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')

    # Plot the image
    img_shape = plot_images(filenames[0], class_name, data_directory)

    # Print the dimensions of the image
    print(f'Image dimensions: {img_shape}')


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