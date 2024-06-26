import os
import json
from sklearn.model_selection import train_test_split

# 데이터셋 경로 설정
dataset_path = 'datasets/kaist-rgbt/train-all-04.txt'
train_txt_path = 'datasets/kaist-rgbt/train.txt'
val_txt_path = 'datasets/kaist-rgbt/val.txt'
output_json = 'utils/eval/KAIST_annotation.json'

# 데이터셋 로드
with open(dataset_path, 'r') as file:
    data = file.readlines()

# 데이터셋을 train과 validation 세트로 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 새로운 분할 데이터를 저장
with open(train_txt_path, 'w') as file:
    file.writelines(train_data)

with open(val_txt_path, 'w') as file:
    file.writelines(val_data)

print(f"Train and validation sets created:\nTrain set: {train_txt_path}\nValidation set: {val_txt_path}")

# KAIST_annotation.json 파일 생성 함수
def create_kaist_annotation_json(val_lines, output_json):
    annotations = []
    
    for index, line in enumerate(val_lines):
        parts = line.strip().split()
        
        # 디버그: 라인 출력
        print(f"Processing line: {line.strip()}")
        
        # 데이터 형식 검증
        if len(parts) < 6:
            print(f"Skipping invalid line: {line.strip()}")
            continue
        
        try:
            image_name = parts[0]
            category_id = int(parts[1])
            bbox = list(map(float, parts[2:6]))
            score = float(parts[6]) if len(parts) > 6 else 1.0  # score 값이 없으면 기본값 1.0 사용
            
            annotation = {
                "image_name": image_name,
                "image_id": index,
                "category_id": category_id,
                "bbox": bbox,
                "score": score
            }
            
            annotations.append(annotation)
        except ValueError as e:
            print(f"Skipping invalid line due to ValueError: {line.strip()} - {e}")
    
    with open(output_json, 'w') as json_file:
        json.dump(annotations, json_file, indent=2)

# val.txt 파일을 기반으로 KAIST_annotation.json 파일 생성
create_kaist_annotation_json(val_data, output_json)
print(f"KAIST_annotation.json created with {len(val_data)} validation entries.")
