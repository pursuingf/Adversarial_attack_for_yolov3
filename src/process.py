import os

def rename_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith("_fail.png"):
            new_name = filename.replace("_fail", "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
            print(f"Renamed: {filename} -> {new_name}")

# Assuming the directory is 'adv_images_v5'
directory_path = 'adv_images_v5'
rename_images(directory_path)