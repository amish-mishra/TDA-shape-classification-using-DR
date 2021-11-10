import numpy as np
import matplotlib.pyplot as plt
import tadasets

pts = 350
d = 3
vector_list = tadasets.dsphere(n=pts, d=d-1, r=1, noise=0.1)


fig = plt.figure()
plt.axis('equal')

#Plot the graph if points are in less than 3 dimensions
if (d==3):
    x = list()
    y = list()
    z = list()
    for v in vector_list:
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
    #3D Plotting
    ax = plt.axes(projection="3d")
    ax.scatter(x,y,z)
if (d==2):
    x = list()
    y = list()
    for v in vector_list:
        x.append(v[0])
        y.append(v[1])
    plt.scatter(x,y)
if (d==1):
    x = list()
    y = np.zeros(len(vector_list))
    for v in vector_list:
        x.append(v[0])
    plt.scatter(x,y)
plt.title("Plot of "+str(pts)+" points")
plt.show()
