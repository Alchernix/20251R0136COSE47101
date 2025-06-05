import pandas as pd

# 엑셀 파일 경로 (파일명.xlsx 부분을 실제 파일명으로 바꾸세요)
file_path = 'data.xlsx'

# C~J 열만 읽기
df = pd.read_excel(file_path, skiprows=1, header=None, usecols='C:J', engine='openpyxl')

subject_set = set()

# 셀 하나씩 접근
for row in df.itertuples(index=False):
    for cell in row:
        if isinstance(cell, str):  # 셀이 문자열일 때만 처리
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subject_set.update(subjects)

# 정렬된 전체 과목 목록
all_subjects = sorted(subject_set)

# 출력 or 저장
for subject in all_subjects:
    print(subject)

# # DataFrame 확인
# print(df.head())  # 첫 5행 출력
# with open('결과.txt', 'w', encoding='utf-8') as f:
#     f.write(df.to_string())