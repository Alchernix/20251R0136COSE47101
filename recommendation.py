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
# 사용자별 수강 과목 벡터 생성 - 가중치 없는 버전
# user_vectors = []

# for row in df.itertuples(index=False):
#     subjects_taken = set()
    
#     for cell in row:
#         if isinstance(cell, str):
#             subjects = [s.strip() for s in cell.split(',') if s.strip()]
#             subjects_taken.update(subjects)
    
#     # 각 과목이 있으면 1, 없으면 0으로 벡터화
#     vector = [1 if subject in subjects_taken else 0 for subject in all_subjects]
#     user_vectors.append(vector)

# vector_df = pd.DataFrame(user_vectors, columns=all_subjects)
#--------------------------------------------------------------------
# 사용자별 수강 과목 벡터 생성 - 최근에 들은 과목일수록 가중치 부여한 버전
user_vectors = []

for row in df.itertuples(index=False):
    # (1) 이 학생이 수강한 학기(셀) 인덱스(컬럼 번호) 모으기
    non_null_semesters = [i for i, cell in enumerate(row) if isinstance(cell, str)]
    k = len(non_null_semesters)   # 이 학생이 수강한 총 학기 수
    
    # (2) 빈 벡터(과목 수 만큼) 생성 (float)
    vec = [0.0] * len(all_subjects)
    
    # (3) 각 학기별로 가중치 부여 → 과목마다 weight 더하기
    #     i번째 non-null 학기 → weight = (i+1) / k
    for idx_in_list, col_idx in enumerate(non_null_semesters):
        weight = (idx_in_list + 1) / k
        cell = row[col_idx]
        
        # 쉼표로 분리된 과목 문자열 → 과목마다 strip
        subjects = [s.strip() for s in cell.split(',') if s.strip()]
        for subj in subjects:
            if subj in all_subjects:
                subj_index = all_subjects.index(subj)
                vec[subj_index] += weight
    
    user_vectors.append(vec)

# 4. DataFrame으로 변환
vector_df = pd.DataFrame(user_vectors, columns=all_subjects)

# 5. 엑셀로 저장
vector_df.to_excel('weighted_user_vectors.xlsx', index=False, engine='openpyxl')
# ===================================================================
# 추천받을 사용자의 수강 과목 벡터 생성 - 가중치 없는 버전
# subjects_taken = set()

# for row in user_df.itertuples(index=False):
#     for cell in row:
#         if isinstance(cell, str):
#             subjects = [s.strip() for s in cell.split(',') if s.strip()]
#             subjects_taken.update(subjects)

# # binary 벡터화
# user_vector = [1 if subject in subjects_taken else 0 for subject in all_subjects]

# user_vector_df = pd.DataFrame([user_vector], columns=all_subjects)

# # 확인
# print(user_vector_df.head())
# user_vector_df.to_excel('user_vector.xlsx', index=False, engine='openpyxl')
#-------------------------------------------------------------------
# 추천받을 사용자의 수강 벡터 생성 - 최근에 들은 과목일수록 가중치 부여한 버전
row = next(user_df.itertuples(index=False))  # 첫 번째(유일한) 사용자의 튜플

# (2) 이 사용자가 과목을 들은 ‘학기 인덱스’(컬럼 번호) 수집
non_null_semesters = [i for i, cell in enumerate(row) if isinstance(cell, str)]
k = len(non_null_semesters)   # 이 사용자가 과목을 수강한 총 학기 수

# (3) 과목별 가중치를 담기 위한 벡터 초기화 (float 타입)
user_weighted_vec = [0.0] * len(all_subjects)

# (4) 각 학기마다 (가중치) 부여 → 그 학기 과목에 누적
for idx_in_list, col_idx in enumerate(non_null_semesters):
    weight = (idx_in_list + 1) / k     # 예: k=3일 때, 학기 순서대로 1/3, 2/3, 3/3
    cell = row[col_idx]                # 해당 학기 셀(쉼표로 구분된 과목 문자열)

    subjects = [s.strip() for s in cell.split(',') if s.strip()]
    for subj in subjects:
        if subj in all_subjects:
            subj_index = all_subjects.index(subj)
            user_weighted_vec[subj_index] += weight

# (5) DataFrame으로 변환
user_vector_df = pd.DataFrame([user_weighted_vec], columns=all_subjects)

# 필요하다면 엑셀로 저장
user_vector_df.to_excel('user_input_weighted_vector.xlsx', index=False, engine='openpyxl')

# ===================================================================
# 코사인 유사도 계산
similarities = cosine_similarity(user_vector_df, vector_df)[0]

similarity_df = pd.DataFrame({
    'user_index': vector_df.index,
    'similarity': similarities
})

similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

# print(similarity_df.head(10).to_string(index=False))
# ===================================================================
# 나와 유사한 n명이 들은 과목 추천
# 유사도 threshold = 0.5
top_users = similarity_df[similarity_df['similarity'] >= 0.5]
top_users_vectors = vector_df.iloc[top_users['user_index']]

# 내가 아직 안 들은 과목 (user_vector에서 0인 과목 인덱스)
user_vector = user_vector_df.iloc[0].values  # Series → numpy array
not_taken_indices = [i for i, val in enumerate(user_vector) if val == 0]
not_taken_subjects = [all_subjects[i] for i in not_taken_indices]

# 누가 어떤 과목을 들었는지 count
recommend_scores = {}

for subject in not_taken_subjects:
    # 해당 과목에 부여된 가중치 총합(소수 포함)
    score = top_users_vectors[subject].sum()
    if score > 0:
        recommend_scores[subject] = score

# 가중치 합계 기준으로 정렬
sorted_recommendations = sorted(recommend_scores.items(), key=lambda x: x[1], reverse=True)

print("추천 과목 (가중치 합계 기준):\n")
for subject, score in sorted_recommendations:
    print(f"{subject}: {score:.2f}점")