import numpy as np
import pandas as pd

student = ["Amit", "Rahul", "Neha", "Priya", "Arjun", "Sneha", "Karan", "Riya"]
subjects=["Maths","Science","English"]

row=len(student)
col=len(subjects)

dataset=np.random.randint(0,100,size=(row,col))
table = pd.DataFrame(dataset, index=student, columns=subjects)

print(table,"\n")

# print("\nMean:\n",table.mean())
# print("\nMedian:\n",table.median())
# print("\nStandard Deviation:\n",table.std())
# print("\nMax value:\n",table.max())
# print("\nMin value:\n",table.min())

print(table.describe())

min_val=table.min()
max_val=table.max()

normalized = (table - min_val) / (max_val - min_val)          #Useful for machine learning models.

print("\nNormalized Marks Table:\n")
print(normalized)

matrix = table.values   # convert DataFrame to NumPy matrix

# Transpose
transpose_matrix = matrix.T

# Generate second matrix
matrix2 = np.random.randint(0, 100, size=matrix.shape)

# Matrix Addition
addition = matrix + matrix2

# Dot Product
dot_product = np.dot(matrix, transpose_matrix)

print("\nTranspose Matrix:\n", transpose_matrix)
print("\nMatrix Addition:\n", addition)
print("\nDot Product Matrix:\n", dot_product)