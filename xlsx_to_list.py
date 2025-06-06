import pandas as pd

file_path = "20251R0136COSE47101/data.xlsx"
# file_path = "/data.xlsx"

df = pd.read_excel(file_path, header=None)

# Columns C to J, Rows 2 to 44
lectures = [[{} for row in range(43)] for col in range(8)]


for col in range(2, 10):  # C~J
    for row in range(1, 44):  # Rows 2 to 44
        cell_value = df.iloc[row, col]
        if pd.notna(cell_value):
            items = {item.strip() for item in str(cell_value).split(',')}
            lectures[col-2][row-1] = items

for col in range(8):
    for row in range(43):
        if lectures[col][row]: print('lectures['+ str(col) + '][' + str(row) + ']' + str(lectures[col][row]))