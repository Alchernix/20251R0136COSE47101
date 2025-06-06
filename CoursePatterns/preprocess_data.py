def get_preprocessed_data(file_path='raw_data.xlsx', mapping_path='course_name.json'):
    import pandas as pd
    import json

    df = pd.read_excel(file_path)

    df_selected = df.iloc[:, list(range(1, 10))].copy()
    df_selected.columns = ["CS_Major", "Year1_Sem1_Subjects", "Year1_Sem2_Subjects",
                           "Year2_Sem1_Subjects", "Year2_Sem2_Subjects",
                           "Year3_Sem1_Subjects", "Year3_Sem2_Subjects",
                           "Year4_Sem1_Subjects", "Year4_Sem2_Subjects"]

    df_selected["CS_Major"] = df_selected["CS_Major"].replace({
        "컴퓨터학과 본전공생이다.": "Primary",
        "컴퓨터학과 복수전공, 이중전공, 융합전공, 부전공생이다.": "NonPrimary"
    })

    with open(mapping_path, "r", encoding="utf-8") as f:
        subject_to_code = json.load(f)

    for col in df_selected.columns[1:]:
        df_selected[col] = df_selected[col].fillna('').astype(str)
        df_selected[col] = df_selected[col].apply(
            lambda x: [subject_to_code.get(s.strip(), s.strip()) for s in x.split(',') if s.strip()]
        )

    return df_selected
