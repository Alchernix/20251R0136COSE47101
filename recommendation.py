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

similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
# ===================================================================
# 나와 유사한 n명이 들은 과목 추천
# 유사도 threshold = 0.6
top_users = similarity_df[similarity_df['similarity'] >= 0.6]
top_users_vectors = vector_df.iloc[top_users['user_index']]

# 내가 아직 안 들은 과목 (user_vector에서 0인 과목 인덱스)
user_vector = user_vector_df.iloc[0].values  # Series → numpy array
not_taken_indices = [i for i, val in enumerate(user_vector) if val == 0]
not_taken_subjects = [all_subjects[i] for i in not_taken_indices]

# 누가 어떤 과목을 들었는지 count
recommend_counts = {}

for subject in not_taken_subjects:
    count = top_users_vectors[subject].sum()
    if count > 0:
        recommend_counts[subject] = count

# 추천 과목 정렬 (많이 들은 순)
sorted_recommendations = sorted(recommend_counts.items(), key=lambda x: x[1], reverse=True)

print("추천 과목:\n")
for subject, count in sorted_recommendations:
    print(f"{subject}: {count}명 수강")