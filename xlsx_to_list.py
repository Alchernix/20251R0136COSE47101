import pandas as pd

file_path = "data.xlsx"
df = pd.read_excel(file_path, header=None)

students = 43

# Define the target cell range: Columns C (index 2) to J (index 9), Rows 2 to 44 (index 1 to 43)
lectures = [[{} for row in range(students)] for col in range(8)]


for col in range(2, 10):  # C~J
    for row in range(1, students+1):  # Rows 2 to 44
        cell_value = df.iloc[row, col]
        if pd.notna(cell_value):
            items = {item.strip() for item in str(cell_value).split(',')}
            lectures[col-2][row-1] = items

# for col in range(8):
#     for row in range(students):
#         if lectures[col][row]: print('lectures['+ str(col) + '][' + str(row) + ']' + str(lectures[col][row]))


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

df = pd.DataFrame(dataframe)

# print(df)
# print(dataframe)

total_courses = set()

for semester in dataframe:
        if semester == 'student_ID': continue
        for stu_no in range(students):
            total_courses.update(dataframe[semester][stu_no])

first_year_cols = ['1y_1s', '1y_2s']
second_year_cols = ['2y_1s', '2y_2s']
third_year_cols = ['3y_1s', '3y_2s']
fourth_year_cols = ['4y_1s', '4y_2s']

df['first_year_courses'] = df[first_year_cols].apply(
    lambda row: set().union(*row), axis=1
)

df['second_year_courses'] = df[second_year_cols].apply(
    lambda row: set().union(*row), axis=1
)

df['third_year_courses'] = df[third_year_cols].apply(
    lambda row: set().union(*row), axis=1
)

df['fourth_year_courses'] = df[fourth_year_cols].apply(
    lambda row: set().union(*row), axis=1
)

result = {}

for course in total_courses:
    # Filter who took 'course' in first grade
    mask = df['first_year_courses'].apply(lambda x: course in x)
    students_with_course = df[mask]
    
    if students_with_course.empty:
        continue
    
    total = len(students_with_course)
    
    # Find the subjects that filtered students took in second grade
    course_counter = {}
    for second_courses in students_with_course['second_year_courses']:
        for c in second_courses:
            if c in total_courses:
                course_counter[c] = course_counter.get(c, 0) + 1
    
    # minsup = 3, min_coef = 0.5
    for next_course, count in course_counter.items():
        ratio = count / total
        if ratio >= 0.5 and count >= 3:
            result[(course, next_course)] = round(ratio, 3)

# Result: 1st grade -> 2nd grade
for (first, next_course), ratio in result.items():
    print("1학년->2학년: " + f"'{first}'를 들은 학생 중 {ratio*100:.1f}%가 2학년 때 '{next_course}'를 수강함.")
