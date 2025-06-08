import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_course import course_df, tag_cols
from preprocess_user import semester_cols
from user_vector import build_user_vector

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
    sims = cosine_similarity(cand_vecs, uvec.reshape(1, -1)).reshape(-1)
    candidates['cos_sim'] = sims

    # ===== 선수관계(prerequisites) =====
    prerequisites = [
        # ===== 본전공 직전학기 데이터 =====
        ('자료구조', '알고리즘', 0.75),      # 본전공 1y_2s → 2y_1s
        ('자료구조', '계산이론', 1.0),       # 본전공 1y_2s → 2y_1s
        ('자료구조', '논리설계', 0.75),      # 본전공 1y_2s → 2y_1s
        ('논리설계', '알고리즘', 0.529),     # 본전공 2y_1s → 2y_2s
        ('논리설계', '컴퓨터구조', 0.824),   # 본전공 2y_1s → 2y_2s
        ('계산이론', '컴퓨터구조', 0.652),   # 본전공 2y_1s → 2y_2s
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
        ('자료구조', '운영체제', 0.75),      # 전체 2y_2s → 3y_1s
        ('공학수학', '컴퓨터네트워크', 0.5), # 전체 2y_2s → 3y_1s
        ('공학수학', '인공지능', 0.5),       # 전체 2y_2s → 3y_1s
        ('전자기학', '데이터베이스', 0.5),   # 전체 2y_2s → 3y_1s
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
        if prereq in taken:
            candidates.loc[
                candidates['Course_Name'] == next_course,
                'cos_sim'
            ] *= (1 + coeff * 0.1)

    return candidates.nlargest(N, 'cos_sim')[
        ['Course_Code','Course_Name','cos_sim']
    ].reset_index(drop=True)