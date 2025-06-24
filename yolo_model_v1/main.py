from create_thermal_ceiling import *
from make_thermal_spot import *
from parse_script import *


def calculate_image_counts(total_images):
    test_count = max(1, int(total_images * 0.10))
    train_val_count = total_images - test_count   
    
    return train_val_count, test_count


if __name__ == "__main__":
    s = int(input("Create images (enter 0) or parse data (enter 1): "))
    
    # Creates data
    if s == 0:
        # Placeholder value 200 can be changed v1 and v2 can replace 149 and 18 below to get a 70/20/10 train/val/test split
        total_images = 200
        v1, v2 = calculate_image_counts(total_images)

        # ~90% Training/Validation Data (~70% Training, ~20% Validation)
        main_create_thermal_ceiling(149, "thermal_data")
        main_make_thermal_spot("thermal_data", 'thermal_with_gradient_spots')

        # ~10% Testing Data
        main_create_thermal_ceiling(18, "test_images")
        main_make_thermal_spot("test_images", "test_images")
    
    # Parses the script to create the training and validation split
    if s == 1:
        main_parse_script()