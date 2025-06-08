import numpy as np
from preprocess_course import course_df, tag_cols
from preprocess_user import semester_cols
from tag_relations import related_tags

def build_user_vector(user_row, course_df, tag_cols, semester_cols,
                      tag_adjustment=True, time_weight=True, w=1.2, bonus=0.1):
    # taken_courses
    taken = user_row.get('taken_courses', [])
    if not isinstance(taken, list) or not taken:
        return np.zeros(len(tag_cols), dtype=float)

    # 과목 매칭
    matched = course_df[course_df['Course_Name'].isin(taken)]
    if matched.empty:
        return np.zeros(len(tag_cols), dtype=float)

    # 태그 합산 → 평균
    sum_vec = matched[tag_cols].sum(axis=0).astype(float).values
    avg_vec = sum_vec / matched.shape[0]

    # 태그별 조정
    if tag_adjustment:
        total_tag_count = course_df[tag_cols].sum(axis=0).astype(float).values
        ratios = np.where(total_tag_count > 0,
                          sum_vec / total_tag_count,
                          0.0)
        avg_vec = avg_vec * ratios

    # 시간 가중치
    if time_weight and semester_cols:
        last_taken = []
        for col in reversed(semester_cols):
            val = user_row.get(col, "")
            if isinstance(val, str) and val.strip():
                last_taken = [x.strip() for x in val.split(',') if x.strip()]
                break
        if last_taken:
            last_matched = course_df[course_df['Course_Name'].isin(last_taken)]
            if not last_matched.empty:
                last_sum = last_matched[tag_cols].sum(axis=0).astype(float).values
                last_avg = last_sum / last_matched.shape[0]
                avg_vec = (avg_vec + last_avg * w) / (1 + w)

    # 태그 간 연관 가중치
    for tag, rels in related_tags.items():
        i = tag_cols.index(tag)
        if avg_vec[i] > 0:
            for rel in rels:
                j = tag_cols.index(rel)
                avg_vec[j] += bonus

    # 최종 정규화
    total = avg_vec.sum()
    if total > 0:
        avg_vec = avg_vec / total

    return avg_vec