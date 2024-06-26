# 파일 내용을 불러옵니다
with open("train.txt", "r") as file:
    lines = file.readlines()

# 줄들을 오름차순으로 정렬합니다
lines.sort()

# 정렬된 줄들을 파일에 다시 저장합니다
with open("train.txt", "w") as file:
    file.writelines(lines)