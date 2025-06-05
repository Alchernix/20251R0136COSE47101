import pandas as pd

# 엑셀 파일 경로 (파일명.xlsx 부분을 실제 파일명으로 바꾸세요)
file_path = 'data.xlsx'

# C~J 열만 읽기
df = pd.read_excel(file_path, skiprows=1, header=None, usecols='C:J', engine='openpyxl')

# ===================================================================
# 전체 과목 목록 추출+정렬
subject_set = set()

for row in df.itertuples(index=False):
    for cell in row:
        if isinstance(cell, str):  # 셀이 문자열일 때만 처리
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subject_set.update(subjects)

all_subjects = sorted(subject_set)
for subject in all_subjects:
    print(subject)

# ===================================================================
# 3. 사용자별 수강 과목 벡터 만들기
user_vectors = []

for row in df.itertuples(index=False):
    subjects_taken = set()
    
    for cell in row:
        if isinstance(cell, str):
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subjects_taken.update(subjects)
    
    # 각 과목이 있으면 1, 없으면 0으로 벡터화
    vector = [1 if subject in subjects_taken else 0 for subject in all_subjects]
    user_vectors.append(vector)

# 4. 결과를 DataFrame으로 변환
vector_df = pd.DataFrame(user_vectors, columns=all_subjects)

# 확인
print(vector_df.head())
vector_df.to_excel('user_course_vectors.xlsx', index=False, engine='openpyxl')