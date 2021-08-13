import numpy as np
import matplotlib.pyplot as plt
import cechmate as cm
import math
from persim import plot_diagrams
import tadasets


x = 0
if x > 0:
    filtration = [([0], 0),
                ([1], 0),
                ([2], 0),
                ([3], 0),
                ([1, 3], math.sqrt(1-x+x**2)),
                ([2, 3], math.sqrt(1-x+x**2)),
                ([0, 1], math.sqrt(3)),
                ([0, 2], math.sqrt(3)),
                ([0, 3], 2-x),
                ([0, 1, 3], 2-x),
                ([0, 3, 2], 2-x)
                ]

elif x < 0:
    filtration = [([0], 0),
                ([1], 0),
                ([2], 0),
                ([3], 0),
                ([1, 3], math.sqrt(1-x+x**2)),
                ([2, 3], math.sqrt(1-x+x**2)),
                ([0, 1], math.sqrt(3)),
                ([0, 2], math.sqrt(3)),
                ([1, 2], math.sqrt(3)),
                ([0, 1, 2], math.sqrt(3)),
                ([1, 2, 3], max(math.sqrt(3), math.sqrt(1-x+x**2)))
                ]

elif x == 0.0:
    filtration = [([0], 0),
                ([1], 0),
                ([2], 0),
                ([3], 0),
                ([0, 1], math.sqrt(3)),
                ([0, 2], math.sqrt(3)),
                ([1, 2], math.sqrt(3)),
                ([1, 3], math.sqrt(1-x+x**2)),
                ([2, 3], math.sqrt(1-x+x**2)),
                ([0, 3], 2-x),
                ([0, 1, 2], math.sqrt(3)),
                ([1, 2, 3], max(math.sqrt(3), math.sqrt(1-x+x**2))),
                ([0, 1, 3], 2-x),               
                ([0, 3, 2], 2-x),
                ]

#Compute persistence diagrams
dgms = cm.phat_diagrams(filtration, show_inf = True)
# print("H0:\n", dgms[0])
# print("H1:\n", dgms[1])

plt.figure()
plt.axis('square')
plot_diagrams(dgms)
# plt.title("Rips Persistence Diagrams")
plt.tight_layout()
plt.show()

