import numpy as np
import matplotlib.pyplot as plt

test_array = [14, 15, 18, 13.5, 14.2, 17, 18, 22]
markers = [3, 6]
markerz = [2, 4]


plt.plot(test_array)
plt.plot(markers, [test_array[i] for i in markers], marker='o', markerfacecolor='red', markeredgecolor='red', linestyle='None')
plt.plot(markerz, [test_array[i] for i in markerz], marker='o', markerfacecolor='green', markeredgecolor='green', linestyle='None')


plt.show()