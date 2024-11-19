# Import necessary libraries# Import necessary libraries
import os
import json

# Function to check if the classes exist in the raw directory
def does_not_exist(folder, directory):

    # Check if the folder exists
    if not os.path.exists(os.path.join(directory, folder)):
        return True
    
    else:
        return False


# Function to return the folders that do not exist
def return_folder_not_exist(subcategories, category, directory):

    # Loop through all the categories
    for subcategory in subcategories:

        # If the category does not exist in the directory, print a message
        if does_not_exist(subcategory, directory):
            print(f'{subcategory} does not exist.')
    
    print(f'{category} class check done!')


# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    images_data_directory = os.path.join(current_directory, 'data', 'images')
    
    # Create an empty dictionary labelled classes
    categories = {}

    # Add the following subcategories to the vehicles class
    category = 'vehicles'
    categories[category] = ['bear', 'bee', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 
                              'dog', 'dolphin', 'duck', 'elephant', 'fish', 'frog', 'giraffe', 'hedgehog', 
                              'kangaroo', 'lion', 'lobster', 'monkey', 'mouse', 'octopus', 'owl', 'panda', 
                              'parrot', 'penguin', 'pig', 'pigeon', 'rabbit', 'rooster', 'scorpion', 'shark', 
                              'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'zebra']
    
    # Check if all of these subcategories exist in the raw file directory
    return_folder_not_exist(categories[category], category, images_data_directory)

    # Add the following subcategories to the animals class
    categories['animals'] = ['airplane', 'blimp', 'bus', 'car', 'pickup truck', 'race car', 'suv', 
                        'truck', 'van', 'bulldozer', 'bicycle', 'motorbike', 'ship', 'speed-boat', 
                        'canoe', 'sailboat', 'submarine', 'helicopter', 'space shuttle', 'train']
    
    # Check if all of these subcategories exist in the raw file directory
    return_folder_not_exist(categories['animals'], 'animals', images_data_directory)
    
    # Add the following subcategories to the food class
    categories['food'] = ['apple', 'banana', 'bread', 'cake', 'carrot', 'donut', 'grapes', 'hamburger', 
                          'hot-dog', 'ice-cream-cone', 'pear', 'pineapple', 'pizza', 'pumpkin', 'strawberry', 'tomato']
    
    # Check if all of these subcategories exist in the raw file directory
    return_folder_not_exist(categories['food'], 'food', images_data_directory)
    
    # Add the following subcategories to the household objects class
    categories['household_objects'] = ['alarm clock', 'armchair', 'ashtray', 'bed', 'bookshelf', 'bowl', 'cabinet', 
                                       'candle', 'chair', 'couch', 'door', 'fan', 'floor lamp', 'fork', 'frying-pan', 
                                       'knife', 'ladder', 'tablelamp', 'mug', 'spoon', 'stapler', 'table', 
                                       'teapot', 'toilet', 'toothbrush', 'tv', 'eyeglasses', 'umbrella']
    
    # Check if all of these subcategories exist in the raw file directory
    return_folder_not_exist(categories['household_objects'], 'household_objects', images_data_directory)
    
    # Add the following categories in the miscellaneous categories
    categories['miscellaneous'] = ['binoculars', 'boomerang', 'brain', 'calculator', 'crown', 'diamond', 'envelope', 'present',
                                    'parachute', 'paper clip', 'human-skeleton', 'sword', 'revolver', 'grenade', 'castle']
    
    # Check if all of these subcategories exist in the raw file directory
    return_folder_not_exist(categories['miscellaneous'], 'miscellaneous', images_data_directory)

    # Save the categories to a .json file
    with open(os.path.join(data_directory, 'categories.json'), 'w') as file:
        json.dump(categories, file, indent=4, sort_keys=True)
    
    
if __name__ == '__main__':

    # Clear the screen regardless of the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

    # Call the main function
    main()