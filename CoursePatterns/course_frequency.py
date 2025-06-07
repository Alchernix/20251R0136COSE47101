def get_course_frequencies():
    from preprocess_data import get_preprocessed_data
    import pandas as pd
    from collections import Counter

    df = get_preprocessed_data()
    all_courses = sum(df[df.columns[1:]].sum(axis=1), [])

    overall_counts = pd.Series(Counter(all_courses)).sort_values(ascending=False)

    primary_courses = sum(df[df["CS_Major"] == "Primary"][df.columns[1:]].sum(axis=1), [])
    non_primary_courses = sum(df[df["CS_Major"] == "NonPrimary"][df.columns[1:]].sum(axis=1), [])

    by_major_counts = pd.DataFrame({
        "Primary": pd.Series(Counter(primary_courses)),
        "NonPrimary": pd.Series(Counter(non_primary_courses))
    }).fillna(0).astype(int).sort_values(by="Primary", ascending=False)

    by_semester_counts = {}
    for col in df.columns[1:]:
        flat = sum(df[col], [])
        by_semester_counts[col] = pd.Series(Counter(flat)).sort_values(ascending=False)

    return overall_counts, by_major_counts, by_semester_counts