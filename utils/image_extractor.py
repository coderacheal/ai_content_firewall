import cv2
import pandas as pd
import os
import glob

# Define the root dataset folder
dataset_root = "Dataset"  # Change this to the path of your Dataset folder
output_folder = "extracted_frames"  # Root folder for extracted images
os.makedirs(output_folder, exist_ok=True)

frame_data = [] 

# Loop through all subdirectories (train, test, val)
for subset in ["train", "test", "val"]:
    subset_path = os.path.join(dataset_root, subset)  # Path to train/test/val folder
    for label in os.listdir(subset_path):  # Loop through labels (e.g., "Adult Content", "Safe Content")
        label_path = os.path.join(subset_path, label)  # Path to each label folder
        if not os.path.isdir(label_path):  # Skip if not a directory
            continue

        # Create the output label folder under extracted_frames
        output_label_folder = os.path.join(output_folder, subset, label)
        os.makedirs(output_label_folder, exist_ok=True)

        # Get all video files in the label folder
        video_files = glob.glob(os.path.join(label_path, "*.mp4")) + \
                      glob.glob(os.path.join(label_path, "*.avi")) + \
                      glob.glob(os.path.join(label_path, "*.mov"))

        # Process each video file
        for video_file in video_files:
            video_name = os.path.splitext(os.path.basename(video_file))[0]  # Video name without extension

            # Load the video
            cap = cv2.VideoCapture(video_file)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate  # Video length in seconds

            count = 0  # Frame counter
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame every 12 seconds
                if count % int(frame_rate * 12) == 0:  
                    timestamp = count / frame_rate  
                    # Save image directly in the label folder
                    image_filename = os.path.join(output_label_folder, f"{video_name}_frame_{count}.jpg")
                    cv2.imwrite(image_filename, frame)  # Save the frame as an image

                    # Append data to list
                    frame_data.append({
                        "Label": label,
                        "Image Path": image_filename
                    })

                count += 1

            # Release video capture
            cap.release()

# Convert to Pandas DataFrame
df = pd.DataFrame(frame_data)

# Save DataFrame to CSV (optional)
df.to_csv("./Dataset/frame_data.csv", index=False)
