import os
import json
from sklearn.model_selection import train_test_split

# Load your dataset
dataset_path = 'datasets/kaist-rgbt/train-all-04.txt'
with open(dataset_path, 'r') as file:
    data = file.readlines()

# Load annotations from annotations.txt
annotations_file_path = 'datasets/kaist-rgbt/annotations/instances_val_annotations.txt'
annotations = {}
with open(annotations_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        image_id = parts[0]
        category_id = int(parts[1])
        bbox = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append({
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],  # width * height
            "iscrowd": 0
        })

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define a function to convert the dataset to KAIST format
def convert_to_kaist(data, annotations, categories):
    images = []
    anns = []
    for idx, line in enumerate(data):
        image_id = line.strip()
        image_info = {
            "id": idx + 1,
            "file_name": image_id,
            "height": 512,  # Replace with actual height
            "width": 640   # Replace with actual width
        }
        images.append(image_info)

        if image_id in annotations:
            for ann in annotations[image_id]:
                ann_info = {
                    "id": len(anns) + 1,
                    "image_id": idx + 1,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                }
                anns.append(ann_info)
    
    kaist_format = {
        "images": images,
        "annotations": anns,
        "categories": categories
    }
    return kaist_format

# Define categories (example, adjust to your actual categories)
categories = [
    {
        "id": 1,
        "name": "person",
        "supercategory": "none"
    },
    {
        "id": 2,
        "name": "cyclist",
        "supercategory": "none"
    },
    {
        "id": 3,
        "name": "people",
        "supercategory": "none"
    },
    {
        "id": 4,
        "name": "person?",
        "supercategory": "none"
    }
]

# Convert train and val data to KAIST format
train_kaist = convert_to_kaist(train_data, annotations, categories)
val_kaist = convert_to_kaist(val_data, annotations, categories)

# Save the new splits in JSON format
train_json_path = 'datasets/kaist-rgbt/annotations/instances_train.json'
val_json_path = 'datasets/kaist-rgbt/annotations/instances_val.json'

os.makedirs(os.path.dirname(train_json_path), exist_ok=True)

with open(train_json_path, 'w') as train_file:
    json.dump(train_kaist, train_file)

with open(val_json_path, 'w') as val_file:
    json.dump(val_kaist, val_file)

print(f"Train and validation JSON files created:\nTrain JSON: {train_json_path}\nValidation JSON: {val_json_path}")
