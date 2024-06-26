import os
import cv2

# 주석 데이터를 저장할 파일 경로
train_annotations_file = 'datasets/kaist-rgbt/annotations/instances_train_annotations.txt'
val_annotations_file = 'datasets/kaist-rgbt/annotations/instances_val_annotations.txt'

# 이미지 및 주석 데이터 경로 (이미지 경로를 실제 경로로 수정)
train_images_dirs = ['datasets/kaist-rgbt/train/images/lwir', 'datasets/kaist-rgbt/train/images/visible']
val_images_dir = 'datasets/kaist-rgbt/val/images'
train_annotations_dir = 'datasets/kaist-rgbt/train/labels'  # 실제 주석 데이터가 저장된 디렉토리
val_annotations_dir = 'datasets/kaist-rgbt/val/labels'  # 실제 주석 데이터가 저장된 디렉토리

# 주석 데이터를 읽어와서 annotations.txt 파일로 저장하는 함수
def create_annotations_file(image_dirs, annotations_dir, output_file):
    with open(output_file, 'w') as f:
        for image_dir in image_dirs:
            for root, _, files in os.walk(image_dir):  # 하위 디렉토리를 포함하여 탐색
                for image_file in files:
                    if image_file.endswith('.jpg'):
                        image_path = os.path.join(root, image_file)
                        annotation_path = os.path.join(annotations_dir, image_file.replace('.jpg', '.txt'))
                        
                        if not os.path.exists(annotation_path):
                            continue
                        
                        with open(annotation_path, 'r') as ann_file:
                            for line in ann_file:
                                parts = line.strip().split()
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # 바운딩 박스 좌표 변환
                                image = cv2.imread(image_path)
                                h, w, _ = image.shape
                                x = int((x_center - width / 2) * w)
                                y = int((y_center - height / 2) * h)
                                bbox_width = int(width * w)
                                bbox_height = int(height * h)
                                
                                # annotations.txt 파일에 작성
                                f.write(f"{os.path.basename(image_path)} {class_id} {x} {y} {bbox_width} {bbox_height}\n")

# 주석 데이터 파일 생성
create_annotations_file(train_images_dirs, train_annotations_dir, train_annotations_file)
print(f"Train annotations file created: {train_annotations_file}")

create_annotations_file([val_images_dir], val_annotations_dir, val_annotations_file)
print(f"Val annotations file created: {val_annotations_file}")
