import cv2
from google.colab.patches import cv2_imshow  # Colab에서 이미지를 표시하기 위한 패키지

# 이미지와 라벨 파일 경로 설정
image_file = 'levle2_51_jpg.rf.6f395bd3f3e96da16305a25f08ef1698.jpg'
label_file = 'levle2_51_jpg.rf.6f395bd3f3e96da16305a25f08ef1698.txt'

# 이미지 로드
image = cv2.imread(image_file)
if image is None:
    print(f"이미지 파일을 로드할 수 없습니다: {image_file}")
    exit()

# 이미지 크기 확인
height, width, _ = image.shape

# 라벨 파일 읽기
try:
    with open(label_file, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"라벨 파일을 찾을 수 없습니다: {label_file}")
    exit()

# 박스 그리기
for line in lines:
    data = line.strip().split()
    class_id, center_x, center_y, box_width, box_height = map(float, data)

    # YOLO 좌표를 픽셀 좌표로 변환
    x1 = int((center_x - box_width / 2) * width)
    y1 = int((center_y - box_height / 2) * height)
    x2 = int((center_x + box_width / 2) * width)
    y2 = int((center_y + box_height / 2) * height)

    # 박스 그리기
    color = (0, 255, 0)  # 초록색
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 클래스 ID 표시
    label_text = f"Class {int(class_id)}"
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 결과 이미지 표시 (Colab에서는 cv2_imshow 사용)
cv2_imshow(image)

# 결과 저장
output_file = 'levle2_51_jpg.rf.6f395bd3f3e96da16305a25f08ef1698.jpg'
cv2.imwrite(output_file, image)
print(f"결과 저장 완료: {output_file}")