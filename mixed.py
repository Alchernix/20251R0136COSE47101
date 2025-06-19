import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

course_tag_path = 'course_tag.xlsx' #태그 파일
df = pd.read_excel('data.xlsx', skiprows=1, header=None, usecols='C:J', engine='openpyxl') # 전체 유저 데이터 파일
raw_user = pd.read_excel('user_data.xlsx', engine='openpyxl')
user_df = pd.read_excel('user_data.xlsx', header=None, skiprows=1, engine='openpyxl')
# 태그 추천=============================================================
# course_tag.xlsx 전처리
raw_course = pd.read_excel(course_tag_path, header=1, skiprows=[2, 3])

# "과목코드" 열 자동 감지
#      - 보통 'COSE' 접두사를 포함하므로, Unnamed* 중에서 COSE 패턴이 있는 칼럼을 찾아냄
possible_code_cols = [
    c for c in raw_course.columns
    if c.startswith('Unnamed') and raw_course[c].dropna().astype(str).str.contains('COSE').any()
]
if not possible_code_cols:
    raise RuntimeError("과목코드 열을 자동으로 감지하지 못했습니다. 컬럼명을 직접 확인해주세요.")
course_code_col = possible_code_cols[0]

# "과목명(태그명)" 열 자동 감지
possible_name_cols = [
    c for c in raw_course.columns
    if '태그명' in c or '과목명' in c or 'Course_Name' in c
]
if not possible_name_cols:
    raise RuntimeError("과목명(태그명) 열을 찾지 못했습니다. 컬럼명을 직접 확인해주세요.")
course_name_col = possible_name_cols[0]

# (칼럼명 변경: 과목코드 → 'Course_Code', 과목명 → 'Course_Name'
raw_course = raw_course.rename(columns={course_code_col: 'Course_Code',
                                        course_name_col: 'Course_Name'})

# 실제 태그 컬럼만 추출
#       - 'Course_Code', 'Course_Name'을 제외한 non-Unnamed 칼럼들을 모두 태그로 감지
tag_cols = [
    c for c in raw_course.columns
    if (not c.startswith('Unnamed')) and (c not in ['Course_Code', 'Course_Name'])
]
# print("추출된 태그 컬럼(tag_cols) 리스트:")
# print(tag_cols, "\n")

# "과목명(Course_Name)"이 NaN인 행 제거
course_df = raw_course[raw_course['Course_Name'].notna()].copy().reset_index(drop=True)

# 필요한 칼럼만 남기기: ['Course_Code', 'Course_Name'] + tag_cols
keep_cols = ['Course_Code', 'Course_Name'] + tag_cols
course_df = course_df[keep_cols].copy()

# 태그 컬럼을 0/1 이진값으로 변환
for t in tag_cols:
    # NaN → 0, 문자열(태그명) → 1
    course_df[t] = course_df[t].notna().astype(int)

#추천받을 유저 파일 전처리
# user_data.xlsx 전처리
#    - “학기에 들은 컴퓨터학과 전공 과목” 칼럼들을 모두 찾아 수강 과목 리스트로 합침

# '학기에 들은' 이라는 문구가 포함된 칼럼명 모두 수집
semester_cols = [c for c in raw_user.columns if '학기에 들은' in c]

# 사용자가 수강한 과목을 모아 리스트로 반환하는 함수 정의
def collect_taken_courses(row):
    taken = []
    for col in semester_cols:
        val = row[col]
        if isinstance(val, str) and val.strip():
            # 쉼표(,) 기준으로 분리하되, 빈 문자열 제외
            courses = [x.strip() for x in val.split(',') if x.strip()]
            taken.extend(courses)
    # 중복 제거
    return list(set(taken))

# 새로운 DataFrame 생성 및 'taken_courses' 컬럼 추가
user_df_processed = raw_user.copy()
user_df_processed['taken_courses'] = user_df_processed.apply(collect_taken_courses, axis=1)

# print("태그 추천 - 추천받을 유저 전처리")
# print(user_df_processed)

#과목 태그간 관계 반영
# 1) tag_cols 정의된 이후, 연관 태그 자동 생성
n_tags = len(tag_cols)
jaccard = np.zeros((n_tags, n_tags), dtype=float)
for i, t1 in enumerate(tag_cols):
    idx1 = set(course_df.index[course_df[t1] == 1])
    for j, t2 in enumerate(tag_cols):
        if i == j:
            continue
        idx2 = set(course_df.index[course_df[t2] == 1])
        union = idx1 | idx2
        inter = idx1 & idx2
        jaccard[i, j] = len(inter) / len(union) if union else 0.0

threshold = 0.35
related_tags = {
    tag_cols[i]: [tag_cols[j] for j in range(n_tags) if jaccard[i, j] >= threshold]
    for i in range(n_tags)
}
# print(related_tags)

#    - course_vectors: numpy array shape = (num_courses, num_tags)
course_vectors = course_df[tag_cols].values.astype(float)  # shape (M, T)   ->  58 x 16 (58과목에 대한 16개 tag들)

# print(course_df[tag_cols].head())   # 58과목에 대한 과목별 태그벡터들 matrix

#사용자 태그 벡터 생성
def build_user_vector(user_row, course_df, tag_cols, semester_cols,
                      tag_adjustment=True, time_weight=True, w=1.2, bonus=0.1):
    # (A) taken_courses 뽑기
    taken = user_row.get('taken_courses', [])
    if not isinstance(taken, list) or not taken:
        return np.zeros(len(tag_cols), dtype=float)

    # (B) 과목 매칭
    matched = course_df[course_df['Course_Name'].isin(taken)]
    if matched.empty:
        return np.zeros(len(tag_cols), dtype=float)

    # (C) 태그 합산→평균
    sum_vec = matched[tag_cols].sum(axis=0).astype(float).values
    avg_vec = sum_vec / matched.shape[0]

    # (D) 태그별 조정
    if tag_adjustment:
        total_tag_count = course_df[tag_cols].sum(axis=0).astype(float).values
        ratios = np.where(total_tag_count > 0,
                          sum_vec / total_tag_count,
                          0.0)   # "전체 태그 과목 중 내가 차지하는 비율"
        avg_vec = avg_vec * ratios

    # (E) 시간 가중치: 실제 수강한 마지막 학기 찾아 적용
    if time_weight and semester_cols:
        last_taken = []
        # semester_cols 리스트를 뒤에서부터 순회하며
        # 비어 있지 않은 첫 번째(=가장 마지막) 학기를 찾는다.
        for col in reversed(semester_cols):
            val = user_row.get(col, "")
            if isinstance(val, str) and val.strip():
                last_taken = [x.strip() for x in val.split(',') if x.strip()]
                #print(last_taken)
                break  # 한 번 찾으면 반복 종료

        # 찾은 마지막 학기 과목이 있을 때만 가중치 적용
        if last_taken:
            last_matched = course_df[course_df['Course_Name'].isin(last_taken)]
            #print(last_matched)
            if not last_matched.empty:
                last_sum = last_matched[tag_cols].sum(axis=0).astype(float).values
                last_avg = last_sum / last_matched.shape[0]
                # 기존 벡터 avg_vec에 w 배수만큼 더해 재정규화
                avg_vec = (avg_vec + last_avg * w) / (1 + w)

    # (F) 태그 간 연관 가중치 부여    ############ 여기서 related_tags(태그 관계) 반영 ###########
    for tag, rels in related_tags.items():
        i = tag_cols.index(tag)
        if avg_vec[i] > 0:
            for rel in rels:
                j = tag_cols.index(rel)
                avg_vec[j] += bonus

    # (G) 최종 정규화
    total = avg_vec.sum()
    if total > 0:
        avg_vec = avg_vec / total

    return avg_vec
#수강기록 기반 추천=========================================================================
# 전체 과목 목록 추출+정렬
subject_set = set()

for row in df.itertuples(index=False):
    for cell in row:
        if isinstance(cell, str):  # 셀이 문자열일 때만 처리
            subjects = [s.strip() for s in cell.split(',') if s.strip()]
            subject_set.update(subjects)

all_subjects = sorted(subject_set)

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

# 코사인 유사도 계산
similarities = cosine_similarity(user_vector_df, vector_df)[0]

similarity_df = pd.DataFrame({
    'user_index': vector_df.index,
    'similarity': similarities
})

similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

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

#선수관계 반영
# 추천 함수
def recommend(user_row, course_df, tag_cols, semester_cols, N=5,
              tag_adjustment=True, time_weight=True, w=1.2, bonus=0.1):
    uvec = build_user_vector(user_row, course_df, tag_cols, semester_cols,
                             tag_adjustment, time_weight, w, bonus)
    if not uvec.any():
        return pd.DataFrame(columns=['Course_Code','Course_Name','cos_sim'])

    taken = user_row.get('taken_courses', [])
    candidates = course_df[~course_df['Course_Name'].isin(taken)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=['Course_Code','Course_Name','cos_sim'])

    cand_vecs = candidates[tag_cols].values.astype(float)
    sims = cosine_similarity(cand_vecs, uvec.reshape(1, -1)).reshape(-1)    ############# 사용자 태그 벡터와, 미리 생성해놓았던 과목들 태그 벡터 간 코사인 유사도 계산

    candidates['cos_sim'] = sims

    # ===== 추가된 선수관계 가중치 적용 부분 =====
    prerequisites = [
        # ===== 본전공 직전학기 데이터 =====
        ('자료구조', '알고리즘', 0.75),      # 본전공 1학년_2학기 → 2학년_1학기
        ('자료구조', '계산이론', 1.0),       # 본전공 1학년_2학기 → 2학년_1학기
        ('자료구조', '논리설계', 0.75),      # 본전공 1학년_2학기 → 2학년_1학기
        ('논리설계', '알고리즘', 0.529),     # 본전공 2학년_1학기 → 2학년_2학기
        ('논리설계', '컴퓨터구조', 0.824),   # 본전공 2학년_1학기 → 2학년_2학기
        ('계산이론', '컴퓨터구조', 0.652),   # 본전공 2학년_1학기 → 2학년_2학기
        ('알고리즘', '컴퓨터구조', 0.538),   # 본전공 2y_1s → 2y_2s
        ('자료구조', '컴퓨터구조', 0.632),   # 본전공 2y_1s → 2y_2s
        ('이산수학', '컴퓨터구조', 0.6),     # 본전공 2y_1s → 2y_2s
        ('공학수학', '운영체제', 0.75),      # 본전공 2y_2s → 3y_1s
        ('데이터베이스', '운영체제', 0.6),   # 본전공 2y_2s → 3y_1s
        ('전자기학', '컴퓨터네트워크', 0.5), # 본전공 2y_2s → 3y_1s
        ('전자기학', '운영체제', 0.667),     # 본전공 2y_2s → 3y_1s
        ('프로그래밍언어', '운영체제', 0.667),# 본전공 2y_2s → 3y_1s
        ('프로그래밍언어', '데이터베이스', 0.667),# 본전공 2y_2s → 3y_1s
        ('알고리즘', '운영체제', 0.636),     # 본전공 2y_2s → 3y_1s
        ('알고리즘', '데이터베이스', 0.545), # 본전공 2y_2s → 3y_1s
        ('컴퓨터네트워크', '운영체제', 0.5), # 본전공 2y_2s → 3y_1s
        ('데이터통신', '운영체제', 0.6),     # 본전공 2y_2s → 3y_1s
        ('데이터통신', '컴퓨터네트워크', 0.6),# 본전공 2y_2s → 3y_1s
        ('데이터통신', '인공지능', 0.8),     # 본전공 2y_2s → 3y_1s
        ('기계학습', '컴퓨터네트워크', 0.75),# 본전공 2y_2s → 3y_1s
        ('기계학습', '인공지능', 0.75),      # 본전공 2y_2s → 3y_1s
        ('확률및랜덤과정', '운영체제', 0.75),# 본전공 2y_2s → 3y_1s
        ('컴퓨터구조', '운영체제', 0.647),   # 본전공 2y_2s → 3y_1s

        # ===== 전체 직전학기 데이터 (본전공에 없는 경우만 추가) =====
        ('자료구조', '운영체제', 0.75),      # 전체 2학년_2학기 → 3학년_1학기
        ('공학수학', '컴퓨터네트워크', 0.5), # 전체 2학년_2학기 → 3학년_1학기
        ('공학수학', '인공지능', 0.5),       # 전체 2학년_2학기 → 3학년_1학기
        ('전자기학', '데이터베이스', 0.5),   # 전체 2학년_2학기 → 3학년_1학기
        ('전자기학', '인공지능', 0.5),       # 전체 2y_2s → 3y_1s
        ('프로그래밍언어', '컴퓨터그래픽스', 0.5),# 전체 2y_2s → 3y_1s
        ('프로그래밍언어', '데이터과학', 0.5), # 전체 2y_2s → 3y_1s
        ('컴퓨터네트워크', '데이터과학', 0.5),# 전체 2y_2s → 3y_1s

        # ===== 연도 단위 데이터 (직전학기에 없는 경우만 보조자료로 추가) =====
        ('자료구조', '인공지능', 0.6),       # 본전공/전체 1학년→2학년
        ('자료구조', '컴퓨터구조', 0.6),     # 본전공/전체 1학년→2학년
        ('기계학습', '딥러닝', 0.75),        # 본전공/전체 2학년→3학년
        ('기계학습', '데이터베이스', 0.75),  # 본전공/전체 2학년→3학년
        ('이산수학', '운영체제', 0.68),      # 본전공/전체 2학년→3학년
        ('공학수학', '신호및시스템', 0.75),  # 본전공/전체 3학년→4학년
        ('기계학습', '자연어처리', 0.75),    # 본전공/전체 3학년→4학년
        ('컴파일러', '자연어처리', 0.75),    # 본전공/이중전공 3학년→4학년
        ('딥러닝', '자연어처리', 0.667),     # 이중전공 3학년→4학년
        ('컴퓨터구조', '딥러닝', 0.556),     # 이중전공 3학년→4학년
    ]
    for prereq, next_course, coeff in prerequisites:
        if prereq in taken:  # taken: 사용자 수강 이력
            candidates.loc[candidates['Course_Name'] == next_course, 'cos_sim'] *= (1 + coeff * 0.1)
            # coeff 값을 바로 쓰는 게 아니라 10% 정도 수준으로 축소된 형태로 사용하여, 선수관계가 과하게 영향을 미치지 않고 자연스럽게 소폭만 추천에 영향을 주도록 조정
    # ==========================================

    return candidates.nlargest(N, 'cos_sim')[['Course_Code','Course_Name','cos_sim']].reset_index(drop=True)

# 결과 출력
rec = recommend(user_df_processed.iloc[0], course_df, tag_cols, semester_cols)
print(f"\n-- 추천 Top-{len(rec)} --")    # def recommend에서 N=5과목으로 설정
print(rec.to_string(index=False))