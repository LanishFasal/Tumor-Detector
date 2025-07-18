import os
import shutil

# Source: where you extracted the dataset
SOURCE_ROOT = r'C:\Users\Pc\Downloads\archive (51)\lgg-mri-segmentation\kaggle_3m'
# Destination: your project data folders
DEST_RAW = r'C:\Users\Pc\Documents\Tumor Detection\raw'
DEST_MASKS = r'C:\Users\Pc\Documents\Tumor Detection\masks'

os.makedirs(DEST_RAW, exist_ok=True)
os.makedirs(DEST_MASKS, exist_ok=True)

for patient_folder in os.listdir(SOURCE_ROOT):
    patient_path = os.path.join(SOURCE_ROOT, patient_folder)
    if os.path.isdir(patient_path):
        for fname in os.listdir(patient_path):
            fpath = os.path.join(patient_path, fname)
            if fname.lower().endswith('.tif'):
                # Build a unique new name: patient_folder + original filename
                new_name = f"{patient_folder}_{fname}"
                if fname.endswith('_mask.tif'):
                    dest = os.path.join(DEST_MASKS, new_name)
                    shutil.copy(fpath, dest)
                    print(f'Copied mask: {fpath} -> {dest}')
                else:
                    dest = os.path.join(DEST_RAW, new_name)
                    shutil.copy(fpath, dest)
                    print(f'Copied image: {fpath} -> {dest}')
print('Done!') 