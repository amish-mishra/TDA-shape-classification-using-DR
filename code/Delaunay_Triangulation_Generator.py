import tadasets
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

    
pts = 50
data = tadasets.infty_sign(n=pts, noise=.2)
tri = Delaunay(data)

fig = plt.figure()
ax = fig.add_subplot()
plt.triplot(data[:,0], data[:,1], tri.simplices.copy())
plt.plot(data[:,0], data[:,1], 'o', color='black')
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()
