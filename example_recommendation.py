from preprocess_course import course_df, tag_cols
from preprocess_user   import user_df_processed, semester_cols
from recommend         import recommend

for idx in range(min(5, len(user_df_processed))):
    rec = recommend(
        user_df_processed.iloc[idx],
        course_df,
        tag_cols,
        semester_cols
    )
    print(f"\n-- 사용자 {idx+1} 추천 Top-{len(rec)} --")
    print(rec.to_string(index=False))