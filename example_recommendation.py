from preprocess_course import course_df, tag_cols
from preprocess_user   import user_df_processed, semester_cols
from recommend         import recommend

# 4) 예시: 상위 5개 추천 결과 출력 
for idx in range(min(5, len(user_df_processed))):
    user_row = user_df_processed.iloc[idx]
    rec_df = recommend(
        user_row,
        course_df,
        tag_cols,
        semester_cols,
        N=5,
        tag_adjustment=True,
        time_weight=True,
        w=1.2
    )
    print(f"\n----- 사용자 {idx+1} 추천 Top-5 (taken_courses = {user_row['taken_courses']}) -----")
    if rec_df.empty:
        print("추천할 과목이 없습니다.")
    else:
        print(rec_df.to_string(index=False))
