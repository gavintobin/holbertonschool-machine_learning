#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


bins = 50

fig, ax = plt.subplots()
n, bins, patches = plt.hist(student_grades, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85)

ax.set_xlabel('Grades')
ax.set_ylabel('Number of Students')
ax.set_title('Project A')

ax.set_edgecolor('black')

plt.show()
