import os
from sklearn.model_selection import train_test_split

# Load your dataset
dataset_path = 'datasets/kaist-rgbt/train-all-04.txt'
with open(dataset_path, 'r') as file:
    data = file.readlines()

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the new splits
train_txt_path = 'datasets/kaist-rgbt/train.txt'
val_txt_path = 'datasets/kaist-rgbt/val.txt'

os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)

with open(train_txt_path, 'w') as file:
    file.writelines(train_data)

with open(val_txt_path, 'w') as file:
    file.writelines(val_data)

print(f"Train and validation sets created:\nTrain set: {train_txt_path}\nValidation set: {val_txt_path}")
