import os
import shutil

folder_a = '/scratch/ssd004/scratch/robzeh/baselines/fid1'
folder_b = '/scratch/ssd004/scratch/robzeh/MusicCaps'
folder_c = '/scratch/ssd004/scratch/robzeh/baselines/mfid1'

# ignore fodlers in folder b
folder_b_list = os.listdir(folder_b)

# Loop through every file in folder_a
for file in os.listdir(folder_a):
    # If the file is a folder, ignore it
    if os.path.isdir(os.path.join(folder_a, file)):
        continue

    file_path_a = os.path.join(folder_a, file)
    
    # filenames are of form id_0.wav, remove _0 from filename
    file = file.replace('_0', '')
    print(file)

    # If the file also exists in folder_b
    if file in folder_b_list:
        # folder ignore
        if file == 'ignore':
            continue

        file_path_b = os.path.join(folder_b, file)

        # Copy the file to folder_c
        shutil.copy(file_path_b, folder_c)
