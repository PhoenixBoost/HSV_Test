import configparser
import os
import cv2
import numpy as np
import random
from collections import Counter
from colorama import Fore, init, Style

# Set up colorama for console output
init(autoreset=True)

def load_config(config_file='config.ini'):
    """
    Load HSV lower and upper values from a config file, or create a new one with default values.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        config['HSV'] = {
            'HSV_Custom_Lower': '30,160,180',
            'HSV_Custom_Upper': '30,255,255'
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)
        print(Fore.GREEN + f"Created {config_file} with default values." + Style.RESET_ALL)
    else:
        config.read(config_file, encoding='utf-8')
    
    hsv_lower_str = config['HSV'].get('HSV_Custom_Lower', '30,160,180')
    hsv_upper_str = config['HSV'].get('HSV_Custom_Upper', '30,255,255')
    hsv_lower = list(map(int, hsv_lower_str.split(',')))
    hsv_upper = list(map(int, hsv_upper_str.split(',')))
    
    return hsv_lower, hsv_upper

def extract_unique_colors(folder_path, hsv_lower, hsv_upper, output_folder):
    """
    Extract unique 'desired' and 'undesired' colors from images in the specified folder.
    Saves the filtered 'desired' images to the output folder.
    """
    ftypes = [".jpg", ".JPG", ".JPEG", ".png", ".PNG", ".gif", ".GIF"]
    unique_desired = set()
    unique_undesired = set()

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if not any(filename.endswith(ext) for ext in ftypes):
            continue

        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        image = cv2.imread(file_path)
        if image is None:
            continue

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))

        # Extract pixel colors within the HSV range
        pixels = hsv_image.reshape(-1, 3)
        for pixel in pixels:
            h, s, v = pixel
            if hsv_lower[0] <= h <= hsv_upper[0] and \
               hsv_lower[1] <= s <= hsv_upper[1] and \
               hsv_lower[2] <= v <= hsv_upper[2]:
                unique_desired.add((int(h), int(s), int(v)))
            else:
                unique_undesired.add((int(h), int(s), int(v)))

        # Save filtered 'desired' image with added accuracy text
        desired_image = cv2.bitwise_and(image, image, mask=mask)

        # Add accuracy text to the image
        add_accuracy_text_to_image(desired_image, unique_desired, unique_undesired)

        output_path = os.path.join(output_folder, f"filtered_desired_{filename}")
        cv2.imwrite(output_path, desired_image)

    return unique_desired, unique_undesired

def add_accuracy_text_to_image(image, unique_desired, unique_undesired):
    """
    Add accuracy text on the image with the accuracy percentage.
    The text size and position are adjusted dynamically based on the image size.
    """
    # Calculate the accuracy (simulated)
    total_desired = len(unique_desired)
    total_undesired = len(unique_undesired)
    total_unique = total_desired + total_undesired
    simulated_accuracy = (total_desired / total_unique * 100) if total_unique > 0 else 0

    accuracy_text = f"Accuracy: {simulated_accuracy:.2f}%"

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Calculate font size based on image size
    font_scale = image_width / 1000  # Adjust this scale factor for better fit
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text
    text_size = cv2.getTextSize(accuracy_text, font, font_scale, 2)[0]
    text_width, text_height = text_size

    # Calculate position to place text in the center
    text_x = (image_width - text_width) // 2
    text_y = (image_height + text_height) // 2

    # Draw text on the image
    cv2.putText(image, accuracy_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

def save_required_undesired_colors(undesired_colors, output_dir):
    """
    Save the required undesired colors based on the calculated accuracy into a text file.
    """
    required_undesired_output_path = os.path.join(output_dir, "required_undesired_colors.txt")
    with open(required_undesired_output_path, "w", encoding="utf-8") as f:
        f.write(f"Undesired colors: {';'.join([f'[{h},{s},{v}]' for (h, s, v) in undesired_colors])};\n")

    print(Fore.GREEN + f"Required undesired colors saved to: {required_undesired_output_path}" + Style.RESET_ALL)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = os.path.join(script_dir, "M")  # Folder containing images
    output_folder = os.path.join(script_dir, "F")  # Folder to save filtered images
    output_dir = os.path.join(script_dir, "Out_Put")  # Folder to save result files

    # Create necessary directories if they don't exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load HSV values from the config file
    hsv_lower, hsv_upper = load_config()
    print(Fore.CYAN + f"Loaded HSV values from config.ini:\n  HSV_Custom_Lower = {hsv_lower}\n  HSV_Custom_Upper = {hsv_upper}" + Style.RESET_ALL)

    # Input the target accuracy for the Aimbot
    target_accuracy = int(input(Fore.YELLOW + "Enter the target accuracy for Aimbot (0-100): " + Style.RESET_ALL))

    # Extract unique colors and calculate accuracy
    unique_desired, unique_undesired = extract_unique_colors(folder_path, hsv_lower, hsv_upper, output_folder)

    total_desired = len(unique_desired)
    total_undesired = len(unique_undesired)
    total_unique = total_desired + total_undesired

    simulated_accuracy = (total_desired / total_unique * 100) if total_unique > 0 else 0
    required_undesired = total_unique * (100 - target_accuracy) / 100

    print(Fore.GREEN + "\nColor Analysis Results:" + Style.RESET_ALL)
    print(f"  Total colors: {total_unique}")
    print(f"  Desired colors (Aimbot working): {total_desired}")
    print(f"  Undesired colors (Aimbot avoiding): {total_undesired}")
    print(f"  Simulated Aimbot accuracy: {simulated_accuracy:.2f}%")
    print(f"  Undesired colors to avoid for target accuracy ({target_accuracy}%): {required_undesired:.2f}")

    # Save results to files
    hsv_colors_output_path = os.path.join(output_dir, "hsv_colors_output.txt")
    with open(hsv_colors_output_path, "w", encoding="utf-8") as f:
        f.write(f"Undesired colors: {';'.join([f'[{h},{s},{v}]' for (h, s, v) in unique_undesired])};\n")
        f.write(f"Desired colors: {';'.join([f'[{h},{s},{v}]' for (h, s, v) in unique_desired])};\n")
        f.write(f"Total unique colors: {total_unique}\n")

    accuracy_output_path = os.path.join(output_dir, "accuracy_output.txt")
    with open(accuracy_output_path, "w", encoding="utf-8") as f:
        f.write(f"Simulated Aimbot accuracy: {simulated_accuracy:.2f}%\n")
        f.write(f"Total colors: {total_unique}\n")
        f.write(f"Desired colors: {total_desired}\n")
        f.write(f"Undesired colors: {total_undesired}\n")
        f.write(f"Undesired colors to avoid for target accuracy ({target_accuracy}%): {required_undesired:.2f}\n")

    # Save required undesired colors based on the calculated amount
    required_count = int(round(required_undesired))
    sampled_undesired = random.sample(list(unique_undesired), required_count) if total_undesired >= required_count and required_count > 0 else list(unique_undesired)

    save_required_undesired_colors(sampled_undesired, output_dir)

    # Filter images based on selected or file-provided undesired colors
    print(Fore.GREEN + "\nResults saved in files:" + Style.RESET_ALL)
    print(f"  {hsv_colors_output_path}")
    print(f"  {accuracy_output_path}")
    print(f"  {os.path.join(output_dir, 'required_undesired_colors.txt')}")

if __name__ == '__main__':
    main()
