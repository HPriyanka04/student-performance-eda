
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
os.makedirs("visualizations", exist_ok=True)


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("student_performance.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nData Types:")
print(data.dtypes)


# ==========================================
# 2. Meaningful Questions and Visualization
# ==========================================

# Question 1: Does study time relate to exam score?

print("\nStudy Hours and Exam Score:")
print(data[["Study_Hours", "Exam_Score"]])

plt.scatter(data["Study_Hours"], data["Exam_Score"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")

plt.savefig("visualizations/study_hours_vs_exam_score.png")

plt.show()


# Question 2: Does attendance relate to exam score?

print("\nAttendance and Exam Score:")
plt.scatter(data["Attendance"], data["Exam_Score"])
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")


plt.savefig("visualizations/attendance_vs_exam_score.png")

plt.show()

# Question 3: Does previous score relate to exam score?

print("\nPrevious Score and Exam Score:")
print(data[["Previous_Score", "Exam_Score"]])

plt.scatter(data["Previous_Score"], data["Exam_Score"])
plt.xlabel("Previous Score")
plt.ylabel("Exam Score")
plt.title("Previous Score vs Exam Score")


plt.savefig("visualizations/previous_score_vs_exam_score.png")
plt.show()


# ==========================================
# 3. Trends and Patterns
# ==========================================

correlation_matrix = data[
    ["Study_Hours", "Attendance", "Previous_Score", "Exam_Score"]
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.to_string())


# ==========================================
# 4. Identify Anomalies
# ==========================================

print("\nMinimum Exam Score:", data["Exam_Score"].min())
print("Maximum Exam Score:", data["Exam_Score"].max())

Q1 = data["Exam_Score"].quantile(0.25)
Q3 = data["Exam_Score"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[
    (data["Exam_Score"] < lower) |
    (data["Exam_Score"] > upper)
]

print("\nExam Score Outliers:")
print(outliers)


# ==========================================
# 5. Hypothesis Testing
# ==========================================

correlation, p_value = pearsonr(
    data["Study_Hours"],
    data["Exam_Score"]
)

print("\nHypothesis Testing:")
print("Correlation:", correlation)
print("P-value:", p_value)


# ==========================================
# 6. Validate Assumptions
# ==========================================

print("\nDescriptive Statistics:")
print(data[["Study_Hours", "Exam_Score"]].describe())


# ==========================================
# 7. Detect Data Issues
# ==========================================

print("\nDuplicate Rows:", data.duplicated().sum())

print("\nMissing Values:")
print(data.isnull().sum())

print(
    "\nInvalid Attendance:",
    ((data["Attendance"] < 0) | (data["Attendance"] > 100)).sum()
)

print(
    "Invalid Study Hours:",
    (data["Study_Hours"] < 0).sum()
)

print(
    "Invalid Previous Scores:",
    ((data["Previous_Score"] < 0) |
     (data["Previous_Score"] > 100)).sum()
)

print(
    "Invalid Exam Scores:",
    ((data["Exam_Score"] < 0) |
     (data["Exam_Score"] > 100)).sum()
)
