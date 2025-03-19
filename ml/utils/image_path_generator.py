import os
import csv

#Map folder names to simpler labels
label_map = {
    "Adult Content": "Adult",
    "Harmful Content": "Harmful",
    "Safe": "Safe",
    "Suicide": "Suicide"
}

def create_image_label_csv(root_dir, output_csv):
    """
    root_dir: The path to your extracted_frames folder.
    output_csv: Where you want the CSV to be created.
    """
    # The class sub-folders in each split
    folder_labels = ["Adult Content", "Harmful Content", "Safe", "Suicide"]
    # The dataset splits
    splits = ["train", "val", "test"]
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Image_path"])
        
        for split in splits:
            for folder_label in folder_labels:
                folder_path = os.path.join(root_dir, split, folder_label)
                if not os.path.isdir(folder_path):
                    continue
                
                # List all image files
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        full_path = os.path.join(folder_path, file_name)
                        
                        # If using label_map:
                        label = label_map.get(folder_label, folder_label)
                        
                        # Otherwise, if you prefer the original folder name:
                        # label = folder_label
                        
                        writer.writerow([label, full_path])
    
    print(f"CSV file created at: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # 1. Point to the extracted_frames folder
    dataset_root = os.path.join("ml", "Dataset", "extracted_frames")
    
    # 2. Point to where you want the CSV to be stored
    output_csv_path = os.path.join("ml", "Dataset", "path_dataset.csv")
    
    create_image_label_csv(dataset_root, output_csv_path)
