#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


cmap = plt.cm.get_cmap('tab10', 4)
cmap.set_over('w')

fig, ax = plt.subplots()


colors = ['red', 'yellow', 'orange', 'peach']
for i, color in enumerate(colors):
    ax.plot([], [], color=cmap(i), label=f'{colors[i]}')

x_labels = ['Farrah', 'Fred', 'Felicia']
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)


for i in range(fruit.shape[1]):
    for j in range(fruit.shape[0]):
        ax.bar(x_labels, fruit[j, i], bottom=0, color=cmap(i), width=0.5)

y_ticks = np.arange(0, 80, 10)
ax.set_yticks(y_ticks)
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')

plt.legend()
plt.show()