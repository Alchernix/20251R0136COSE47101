import pandas as pd

# file_path = "20251R0136COSE47101/data.xlsx"
file_path = "data.xlsx"
df = pd.read_excel(file_path, header=None)

students = 43

# Columns C to J, Rows 2 to {students + 1}
lectures = [[{} for row in range(students)] for col in range(8)]


for col in range(2, 10):  # C~J
    for row in range(1, students+1):  # Rows 2 to {students + 1}
        cell_value = df.iloc[row, col]
        if pd.notna(cell_value):
            items = {item.strip() for item in str(cell_value).split(',')}
            lectures[col-2][row-1] = items

# Code to debug above code (line 13~18)
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

# Code to debug above code (line 26~36)
# print(df)
# print(dataframe)