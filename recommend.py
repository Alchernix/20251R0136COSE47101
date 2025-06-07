from preprocess_course import course_df, tag_cols
from preprocess_user   import semester_cols
from user_vector       import build_user_vector
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 2) 코스 벡터 행렬 미리 추출
#    - course_vectors: numpy array shape = (num_courses, num_tags)
course_vectors = course_df[tag_cols].values.astype(float)  # shape (M, T)


# 3) 추천 함수 정의
#    - 한 명의 user_row → Top-N 추천 결과 DataFrame 반환
def recommend(user_row, course_df, tag_cols, semester_cols, N=5,
              tag_adjustment=True, time_weight=True, w=1.2):
    """
    user_row:       user_df_processed의 한 행 (Series)
    course_df:      전처리된 과목 DataFrame
    tag_cols:       과목 DataFrame에서 태그만 모은 컬럼 리스트
    semester_cols:  user_df_processed에서 “학기에 들은” 칼럼명 리스트
    N:              상위 몇 개 과목을 추천할지
    tag_adjustment: 태그별 조정 적용 여부
    time_weight:    시간 가중치 적용 여부
    w:              마지막 학기 가중치 배수
    """
    # 사용자 벡터 생성 (numpy 1D)
    uvec = build_user_vector(user_row, course_df, tag_cols, semester_cols,
                             tag_adjustment=tag_adjustment,
                             time_weight=time_weight, w=w)
    if np.all(uvec == 0):
        return pd.DataFrame(columns=['Course_Code','Course_Name','cos_sim']) 

    # 사용자가 이미 수강한 과목 제외
    taken = user_row['taken_courses']
    mask = ~course_df['Course_Name'].isin(taken)
    candidate_df = course_df[mask].copy().reset_index(drop=True)
    if candidate_df.shape[0] == 0:
        return pd.DataFrame(columns=['Course_Code','Course_Name','cos_sim'])

    # 후보 과목 벡터들
    cand_vecs = candidate_df[tag_cols].values.astype(float)
    # 코사인 유사도 계산
    sims = cosine_similarity(cand_vecs, uvec.reshape(1, -1)).reshape(-1)
    candidate_df['cos_sim'] = sims

    # 상위 N개 선별
    topN = candidate_df.nlargest(N, 'cos_sim')[['Course_Code','Course_Name','cos_sim']].reset_index(drop=True)
    return topN
