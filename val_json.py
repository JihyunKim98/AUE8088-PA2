import json
import os

# JSON 파일 경로
json_file_path = 'datasets/kaist-rgbt/annotations/KAIST_annotation.json'

# JSON 파일 로드
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 데이터 검증
def validate_data(data):
    if 'images' not in data or 'annotations' not in data:
        return False, "Missing 'images' or 'annotations' key in JSON"
    
    for img in data['images']:
        if 'file_name' not in img or not os.path.exists(img['file_name']):
            return False, f"Image path {img['file_name']} does not exist"
    
    return True, "Validation successful"

# 데이터 검증 실행
is_valid, message = validate_data(data)
print(message)

# 데이터 검증 성공 시 업로드 시도
if is_valid:
    # 여기에 업로드 코드를 추가하세요.
    print("JSON 파일을 업로드할 준비가 되었습니다.")
else:
    print("JSON 파일 검증에 실패했습니다. 파일을 수정하세요.")
