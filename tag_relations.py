import numpy as np
from preprocess_course import course_df, tag_cols

# 태그 간 유사도 계산 (Jaccard)
n_tags = len(tag_cols)
jaccard = np.zeros((n_tags, n_tags), dtype=float)
for i, t1 in enumerate(tag_cols):
    idx1 = set(course_df.index[course_df[t1] == 1])
    for j, t2 in enumerate(tag_cols):
        if i == j: continue
        idx2 = set(course_df.index[course_df[t2] == 1])
        union = idx1 | idx2
        inter = idx1 & idx2
        jaccard[i, j] = len(inter) / len(union) if union else 0.0

threshold = 0.35
related_tags = {
    tag_cols[i]: [tag_cols[j] for j in range(n_tags) if jaccard[i, j] >= threshold]
    for i in range(n_tags)
}