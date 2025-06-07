import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # 1) 사용자 벡터(user profile vector) 생성 함수
#    - 태그별 과목 수 대비 비율 조정(tag_adjustment)
#    - 최근 학기 가중치 부여(time_weight, w 배수)
#    - 반환값은 항상 numpy 1D 배열(shape = (num_tags,))

def build_user_vector(user_row, course_df, tag_cols, semester_cols,
                      tag_adjustment=True, time_weight=True, w=1.2):
    """
    user_row: user_df_processed의 한 행 (Series)
    course_df: ['Course_Code','Course_Name'] + tag_cols(0/1 이진값)으로 전처리된 DataFrame
    tag_cols: course_df에서 실제 태그만 모은 리스트
    semester_cols: user_df_processed에서 '학기에 들은 ...' 칼럼명 리스트
    tag_adjustment: 태그별 비율 조정 적용 여부
    time_weight: 마지막 학기 과목에 가중치 부여 여부
    w: 마지막 학기 가중치 배수 (기본 1.2)

    반환: 길이 len(tag_cols)인 numpy 1D 배열
    """
    # 사용자가 수강한 과목 리스트
    taken = user_row['taken_courses']  # 예: ['자료구조','알고리즘', ...]
    # 만약 taken이 비어 있으면, 0벡터 반환
    if not isinstance(taken, list) or len(taken) == 0:
        return np.zeros(len(tag_cols), dtype=float)

    # course_df와 매칭: 과목명(Course_Name) 기준
    matched = course_df[course_df['Course_Name'].isin(taken)]
    if matched.shape[0] == 0:
        return np.zeros(len(tag_cols), dtype=float)

    # 태그 이진값 합산 → 평균 (numpy)
    sum_vec = matched[tag_cols].sum(axis=0).astype(float).values
    avg_vec = sum_vec / matched.shape[0]

    # 태그별 과목 수 대비 비율 조정 
    if tag_adjustment:
        total_tag_count = course_df[tag_cols].sum(axis=0).astype(float).values
        user_tag_count  = matched[tag_cols].sum(axis=0).astype(float).values
        ratios = np.where(total_tag_count > 0, user_tag_count / total_tag_count, 0.0)
        avg_vec = avg_vec * ratios

    # 마지막 학기 과목 가중치 적용
    if time_weight and semester_cols:
        last_col = semester_cols[-1]
        val = user_row[last_col]
        last_taken = []
        if isinstance(val, str) and val.strip():
            last_taken = [x.strip() for x in val.split(',') if x.strip()]

        if last_taken:
            last_matched = course_df[course_df['Course_Name'].isin(last_taken)]
            if last_matched.shape[0] > 0:
                last_sum = last_matched[tag_cols].sum(axis=0).astype(float).values
                last_avg = last_sum / last_matched.shape[0]
                avg_vec = (avg_vec + last_avg * w) / (1 + w)

    return avg_vec
