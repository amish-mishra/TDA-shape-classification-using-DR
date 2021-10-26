import tadasets
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

    
n_sphere_dim = 1  # the points will be in n+1 dimensions
pts = 70
data = tadasets.dsphere(n=pts, d=n_sphere_dim, r=1, noise=0.1)
tri = Delaunay(data)

fig = plt.figure()
ax = fig.add_subplot()
plt.triplot(data[:,0], data[:,1], tri.simplices.copy())
plt.plot(data[:,0], data[:,1], 'o', color='black')
ax.set_aspect('equal')
plt.title('Delaunay Triangulation of a Noisy Circle')
plt.show()
