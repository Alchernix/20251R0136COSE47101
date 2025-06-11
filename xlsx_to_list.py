import pandas as pd

file_path = "data.xlsx"
df = pd.read_excel(file_path, header=None)

# You can change the variable 'students' if the number of students changes
students = 43

# Define the target cell range: Columns C (index 2) to J (index 9), Rows 2 to 44 (index 1 to 43)
lectures = [[{} for row in range(students)] for col in range(8)]
primary_major = [0 for i in range(students)]

for col in range(2, 10):  # C~J
    for row in range(1, students+1):  # Rows 2 to 44
        cell_value = df.iloc[row, col]
        if pd.notna(cell_value):
            items = {item.strip() for item in str(cell_value).split(',')}
            lectures[col-2][row-1] = items

for i in range(1, students+1):
    if(df.iloc[i, 1] == '컴퓨터학과 본전공생이다.'): primary_major[i-1] = 1

# Code to debug above code (line 13~21)
# for col in range(8):
#     for row in range(students):
#         if lectures[col][row]: print('lectures['+ str(col) + '][' + str(row) + ']' + str(lectures[col][row]))
# print(primary_major)

# Convert list 'lectures' into DataFrame
dataframe = {
    'student_ID': range(students),
    '1y_1s': [lectures[0][i] for i in range(students)],
    '1y_2s': [lectures[1][i] for i in range(students)],
    '2y_1s': [lectures[2][i] for i in range(students)],
    '2y_2s': [lectures[3][i] for i in range(students)],
    '3y_1s': [lectures[4][i] for i in range(students)],
    '3y_2s': [lectures[5][i] for i in range(students)],
    '4y_1s': [lectures[6][i] for i in range(students)],
    '4y_2s': [lectures[7][i] for i in range(students)]
}

dataframe_primary = {
    'student_ID': [i for i in range(students) if primary_major[i] == 1],
    '1y_1s': [lectures[0][i] for i in range(students) if primary_major[i] == 1],
    '1y_2s': [lectures[1][i] for i in range(students) if primary_major[i] == 1],
    '2y_1s': [lectures[2][i] for i in range(students) if primary_major[i] == 1],
    '2y_2s': [lectures[3][i] for i in range(students) if primary_major[i] == 1],
    '3y_1s': [lectures[4][i] for i in range(students) if primary_major[i] == 1],
    '3y_2s': [lectures[5][i] for i in range(students) if primary_major[i] == 1],
    '4y_1s': [lectures[6][i] for i in range(students) if primary_major[i] == 1],
    '4y_2s': [lectures[7][i] for i in range(students) if primary_major[i] == 1]
}

dataframe_not_primary = {
    'student_ID': [i for i in range(students) if primary_major[i] == 0],
    '1y_1s': [lectures[0][i] for i in range(students) if primary_major[i] == 0],
    '1y_2s': [lectures[1][i] for i in range(students) if primary_major[i] == 0],
    '2y_1s': [lectures[2][i] for i in range(students) if primary_major[i] == 0],
    '2y_2s': [lectures[3][i] for i in range(students) if primary_major[i] == 0],
    '3y_1s': [lectures[4][i] for i in range(students) if primary_major[i] == 0],
    '3y_2s': [lectures[5][i] for i in range(students) if primary_major[i] == 0],
    '4y_1s': [lectures[6][i] for i in range(students) if primary_major[i] == 0],
    '4y_2s': [lectures[7][i] for i in range(students) if primary_major[i] == 0]
}

df_all = pd.DataFrame(dataframe)
df_primary = pd.DataFrame(dataframe_primary)
df_not_primary = pd.DataFrame(dataframe_not_primary)

# Code to debug above code (line 66~68)
# print(df)
# print(dataframe)
# print(len(df))

total_courses = set()

# Find all lectures in 'dataframe'
for semester in dataframe:
        if semester == 'student_ID': continue
        for stu_no in range(students):
            total_courses.update(dataframe[semester][stu_no])

first_year_cols = ['1y_1s', '1y_2s']
second_year_cols = ['2y_1s', '2y_2s']
third_year_cols = ['3y_1s', '3y_2s']
fourth_year_cols = ['4y_1s', '4y_2s']

# A function to find sequences of taking courses (on a yearly basis)
# Returns: dict: {(course1, course2): ratio} patterns
def find_course_associations(df, first_year, first_year_cols, second_year_cols, total_courses, min_support=3, min_confidence=0.5):

    result = {}

    for course in total_courses:
        # Filter who took 'course' in first year
        def has_course(row):
            return any(course in row[col] for col in first_year_cols)

        students_with_course = df[df.apply(has_course, axis=1)]

        if students_with_course.empty:
            continue

        total = len(students_with_course)

        # Find the subjects that filtered students took in second year
        course_counter = {}
        for _, row in students_with_course.iterrows():
            second_year_courses = set()
            for col in second_year_cols:
                second_year_courses.update(row[col])
            for c in second_year_courses:
                if c in total_courses:
                    course_counter[c] = course_counter.get(c, 0) + 1

        # Generates patterns
        for next_course, count in course_counter.items():
            ratio = count / total
            if ratio >= min_confidence and count >= min_support:
                result[(course, next_course, count, total)] = round(ratio, 3)

    for (first, next_course, result_count, result_total), ratio in result.items():
        print(str(first_year) + "학년->" + str(first_year + 1) + "학년: " + f"'{first}' 수강생 중 {ratio*100:.1f}%가 다음 해에 '{next_course}'을(를) 수강함({result_count}/{result_total}).")

    return result

# A function to find sequences of taking courses (on a semester-by-semester basis)
# Returns: dict: {(course1, course2): ratio} patterns
def find_semester_course_associations(df, from_semester, to_semester, total_courses, min_support=3, min_confidence=0.5):

    result = {}

    for course in total_courses:
        # Filter who took 'course' in from_semester
        students_with_course = df[df[from_semester].apply(lambda x: course in x)]

        if students_with_course.empty:
            continue

        total = len(students_with_course)

        # Find the subjects that filtered students took in to_semesters
        course_counter = {}
        for next_courses in students_with_course[to_semester]:
            for c in next_courses:
                if c in total_courses:
                    course_counter[c] = course_counter.get(c, 0) + 1

        # Generates patterns
        for next_course, count in course_counter.items():
            ratio = count / total
            if ratio >= min_confidence and count >= min_support:
                result[(course, next_course, count, total)] = round(ratio, 3)

    for (first, next_course, result_count, result_total), ratio in result.items():
        print(f"{from_semester} → {to_semester}: '{first}' 수강생의 {ratio*100:.1f}%가 다음 학기에 '{next_course}'을(를) 수강함({result_count}/{result_total}).")

    return result

# (1) For all students responded in our survey

print("\n\n모든 학생들 대상 분석 결과(minsup = 3, minconf = 0.5)\n")

year_result1to2 = find_course_associations(
    df_all,
    first_year=1,
    first_year_cols=['1y_1s', '1y_2s'],
    second_year_cols=['2y_1s', '2y_2s'],
    total_courses=total_courses
)

year_result2to3 = find_course_associations(
    df_all,
    first_year=2,
    first_year_cols=['2y_1s', '2y_2s'],
    second_year_cols=['3y_1s', '3y_2s'],
    total_courses=total_courses
)

year_result3to4 = find_course_associations(
    df_all,
    first_year=3,
    first_year_cols=['3y_1s', '3y_2s'],
    second_year_cols=['4y_1s', '4y_2s'],
    total_courses=total_courses
)

sem_result1to2 = find_semester_course_associations(df_all, '1y_1s', '1y_2s', total_courses)
sem_result2to3 = find_semester_course_associations(df_all, '1y_2s', '2y_1s', total_courses)
sem_result3to4 = find_semester_course_associations(df_all, '2y_1s', '2y_2s', total_courses)
sem_result4to5 = find_semester_course_associations(df_all, '2y_2s', '3y_1s', total_courses)
sem_result5to6 = find_semester_course_associations(df_all, '3y_1s', '3y_2s', total_courses)
sem_result6to7 = find_semester_course_associations(df_all, '3y_2s', '4y_1s', total_courses)
sem_result7to8 = find_semester_course_associations(df_all, '4y_1s', '4y_2s', total_courses)

# (2) For main major of CSE students responded in our survey

print("\n\n컴퓨터학과 본전공생 대상 분석 결과(minsup = 3, minconf = 0.5)\n")

year_result1to2 = find_course_associations(
    df_primary,
    first_year=1,
    first_year_cols=['1y_1s', '1y_2s'],
    second_year_cols=['2y_1s', '2y_2s'],
    total_courses=total_courses
)

year_result2to3 = find_course_associations(
    df_primary,
    first_year=2,
    first_year_cols=['2y_1s', '2y_2s'],
    second_year_cols=['3y_1s', '3y_2s'],
    total_courses=total_courses
)

year_result3to4 = find_course_associations(
    df_primary,
    first_year=3,
    first_year_cols=['3y_1s', '3y_2s'],
    second_year_cols=['4y_1s', '4y_2s'],
    total_courses=total_courses
)

sem_result1to2 = find_semester_course_associations(df_primary, '1y_1s', '1y_2s', total_courses)
sem_result2to3 = find_semester_course_associations(df_primary, '1y_2s', '2y_1s', total_courses)
sem_result3to4 = find_semester_course_associations(df_primary, '2y_1s', '2y_2s', total_courses)
sem_result4to5 = find_semester_course_associations(df_primary, '2y_2s', '3y_1s', total_courses)
sem_result5to6 = find_semester_course_associations(df_primary, '3y_1s', '3y_2s', total_courses)
sem_result6to7 = find_semester_course_associations(df_primary, '3y_2s', '4y_1s', total_courses)
sem_result7to8 = find_semester_course_associations(df_primary, '4y_1s', '4y_2s', total_courses)

# (3) For double major of CSE students responded in our survey

print("\n\n컴퓨터학과 이중/복수/융합/부전공생 대상 분석 결과(minsup = 3, minconf = 0.5)\n")

year_result1to2 = find_course_associations(
    df_not_primary,
    first_year=1,
    first_year_cols=['1y_1s', '1y_2s'],
    second_year_cols=['2y_1s', '2y_2s'],
    total_courses=total_courses
)

year_result2to3 = find_course_associations(
    df_not_primary,
    first_year=2,
    first_year_cols=['2y_1s', '2y_2s'],
    second_year_cols=['3y_1s', '3y_2s'],
    total_courses=total_courses
)

year_result3to4 = find_course_associations(
    df_not_primary,
    first_year=3,
    first_year_cols=['3y_1s', '3y_2s'],
    second_year_cols=['4y_1s', '4y_2s'],
    total_courses=total_courses
)

sem_result1to2 = find_semester_course_associations(df_not_primary, '1y_1s', '1y_2s', total_courses)
sem_result2to3 = find_semester_course_associations(df_not_primary, '1y_2s', '2y_1s', total_courses)
sem_result3to4 = find_semester_course_associations(df_not_primary, '2y_1s', '2y_2s', total_courses)
sem_result4to5 = find_semester_course_associations(df_not_primary, '2y_2s', '3y_1s', total_courses)
sem_result5to6 = find_semester_course_associations(df_not_primary, '3y_1s', '3y_2s', total_courses)
sem_result6to7 = find_semester_course_associations(df_not_primary, '3y_2s', '4y_1s', total_courses)
sem_result7to8 = find_semester_course_associations(df_not_primary, '4y_1s', '4y_2s', total_courses)