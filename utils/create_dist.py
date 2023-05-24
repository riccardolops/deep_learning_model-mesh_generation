from utils import bin2dist
import torchio as tio
import os
import json

root = '/homes/rlops/Task02_Heart'

with open(os.path.join(root, 'dataset.json')) as dataset_file:
    json_dataset = json.load(dataset_file)

dataset = json_dataset['training']
new_json = []
for patient in dataset:
    patient['distance'] = patient['label'][:-7] + '_dist' + patient['image'][-7:]
    bin2dist(tio.LabelMap(os.path.join(root, patient['label']))).save(os.path.join(root, patient['distance']))
    new_json.append(patient)
    print(patient['distance'] + ' done...')

with open(os.path.join(root, 'dataset_new.json'), "w") as json_file:
    json.dump(new_json, json_file)