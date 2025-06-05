import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 파일 읽기
df = pd.read_excel('data.xlsx', skiprows=1, header=None, usecols='C:J', engine='openpyxl')
# 유저 파일 읽기 - 나중에 방식 수정 예정
user_df = pd.read_excel('user_data.xlsx', header=None, engine='openpyxl')
# ===================================================================
# 전체 과목 목록 추출+정렬
subject_set = set()

for row in df.itertuples(index=False):
    for cell in row:
        if isinstance(cell, str):  # 셀이 문자열일 때만 처리
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subject_set.update(subjects)

all_subjects = sorted(subject_set)
# for subject in all_subjects:
#     print(subject)

# ===================================================================
# 사용자별 수강 과목 벡터 생성
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

vector_df = pd.DataFrame(user_vectors, columns=all_subjects)
# ===================================================================
# 추천받을 사용자의 수강 과목 벡터 생성
subjects_taken = set()

for row in user_df.itertuples(index=False):
    for cell in row:
        if isinstance(cell, str):
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subjects_taken.update(subjects)

# binary 벡터화
user_vector = [1 if subject in subjects_taken else 0 for subject in all_subjects]

user_vector_df = pd.DataFrame([user_vector], columns=all_subjects)

# # 확인
# print(user_vector_df.head())
# user_vector_df.to_excel('user_vector.xlsx', index=False, engine='openpyxl')
# ===================================================================
# 코사인 유사도 계산
similarities = cosine_similarity(user_vector_df, vector_df)[0]

similarity_df = pd.DataFrame({
    'user_index': vector_df.index,
    'similarity': similarities
})

# 유사도 높은 순 정렬
similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

# 확인
print(similarity_df.head(5).to_string(index=False))