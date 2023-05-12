#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 2])


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

ax1.plot(y0)
ax1.set_title('Plot 1', fontsize='x-small')
ax1.set_xlabel('x', fontsize='x-small')
ax1.set_ylabel('y', fontsize='x-small')

ax2.scatter(x1, y1)
ax2.set_title('Plot 2', fontsize='x-small')
ax2.set_xlabel('x', fontsize='x-small')
ax2.set_ylabel('y', fontsize='x-small')

ax3.plot(x2, y2)
ax3.set_title('Plot 3', fontsize='x-small')
ax3.set_xlabel('x', fontsize='x-small')
ax3.set_ylabel('y', fontsize='x-small')

ax4.plot(x3, y31, label='t31')
ax4.plot(x3, y32, label='t32')
ax4.set_title('Plot 4', fontsize='x-small')
ax4.set_xlabel('x', fontsize='x-small')
ax4.set_ylabel('y', fontsize='x-small')
ax4.legend(fontsize='x-small')

ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
ax5.set_title('Plot 5', fontsize='x-small')
ax5.set_xlabel('x', fontsize='x-small')
ax5.set_ylabel('y', fontsize='x-small')

fig.suptitle('All in One', fontsize='x-small')

plt.tight_layout()

plt.show()

