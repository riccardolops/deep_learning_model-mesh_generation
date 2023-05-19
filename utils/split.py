import os
import json
from sklearn.model_selection import train_test_split

split=False

images = []
masks = []
roott = '/homes/rlops/Aortic_Vessel_Tree'
folder_path = '/homes/rlops/Aortic_Vessel_Tree'

for root, _, files in os.walk(folder_path):
    # find .nrrd files
    img_files = [f for f in files if f.endswith('.nrrd')]
    # if there are files
    if len(img_files) > 0:
        # load image
        i = 0
        if img_files[i].endswith('.seg.nrrd'):
            i = 1
        img_path = os.path.join(root, img_files[i])  # only the first file .nrrd
        #img = itk.imread(img_path)
        #img_array = itk.array_view_from_image(img)
        # find .seg.nrrd files
        mask_files = [f for f in files if f.endswith('.seg.nrrd')]
        if len(mask_files) > 0:
            # load mask
            mask_path = os.path.join(root, mask_files[0])  # only the first file .seg.nrrd
            #mask = itk.imread(mask_path)
            #mask_array = itk.array_view_from_image(mask)

            # aggiungi l'immagine e la maschera alla lista del dataset
            folder_path, file_name = os.path.split(img_path)
            _, parent_folder = os.path.split(folder_path)
            _, grandparent_folder = os.path.split(_)
            img_path = os.path.join(grandparent_folder, parent_folder, file_name)
            images.append(img_path)
            folder_path, file_name = os.path.split(mask_path)
            _, parent_folder = os.path.split(folder_path)
            _, grandparent_folder = os.path.split(_)
            mask_path = os.path.join(grandparent_folder, parent_folder, file_name)
            masks.append(mask_path)

if split:
    image_train, image_val, mask_train, mask_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    data = {"training": [], "test": []}

    for img, mask in zip(image_train, mask_train):
        data["training"].append({"image": img, "label": mask})
    for img, mask in zip(image_val, mask_val):
        data["test"].append({"image": img, "label": mask})
else:
    data = {"training": []}
    for img, mask in zip(images, masks):
        data["training"].append({"image": img, "label": mask})

with open(os.path.join(roott, "dataset.json"), "w") as json_file:
    json.dump(data, json_file)
