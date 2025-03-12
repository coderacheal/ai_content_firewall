import os
import csv

def create_image_csv(root_dir, output_csv):
    """
    Creates a CSV file listing image paths and their labels based on subfolder names.

    :param root_dir: Path to the directory containing the web_images subfolders.
    :param output_csv: Name/path for the CSV output file.
    """
    # Collect data in a list before writing to CSV
    data = []
    
    # Loop through each subfolder in web_images
    for label_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label_folder)

        # Skip any files or hidden folders, only process directories
        if not os.path.isdir(folder_path):
            continue

        # For each image in the subfolder, create an entry
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            
            # Ensure it's actually a file (and presumably an image)
            if os.path.isfile(img_path):
                # Append (label, image_path) tuple
                data.append((label_folder, img_path))
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Label', 'Image Path',])
        # Write data rows
        for label, path in data:
            writer.writerow([label, path])

if __name__ == "__main__":
    # Adjust paths as needed
    web_images_dir = "Dataset/web_images"
    output_csv_path = "./Dataset/web_images_data.csv"
    create_image_csv(web_images_dir, output_csv_path)
    print(f"CSV file created at: {output_csv_path}")
