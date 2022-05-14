from matplotlib import pyplot as plt, animation as an
import numpy as np
from matplotlib.patches import Polygon
from persim import plot_diagrams
import cechmate as cm


def connectpoints(pt1, pt2):
    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]
    plt.plot([x1,x2],[y1,y2], color ='k')


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# data = tadasets.dsphere(n=20, d=1, r=1, noise=0.1)

## Hard coded data
# data = np.array([[ 0.9729797, -0.40156498],
#  [ 1.10507727, -0.25586236],
#  [ 0.80864922, 0.65327502],
#  [-0.94087488, -0.48374892],
#  [-0.27104698,  0.93909997],
#  [ 0.94983557 , 0.20914284],
#  [ 0.12948646 , 1.20295278],
#  [ 0.15898131 ,-0.97494943],
#  [-0.90638196 , 0.31881015],
#  [-0.36961883,-0.78303541],
#  [-0.90593916 , 0.40789757],
#  [ 0.72563306 , 0.47243787],
#  [ 0.23914349 , 0.95738457],
#  [ 0.33497857 , 0.96859328],
#  [-0.45240152 , 0.8804465 ],
#  [ 0.65024778 ,-0.81361107],
#  [ 0.35124217, -0.99322087],
#  [-0.21738975, -0.90805497],
#  [ 1.0454358  ,-0.07676256],
#  [-1.05054797 ,-0.45154826]])

data = np.array([[-0.93881648,  0.03598761],
 [-0.52548888 , 0.84836182],
 [ 0.83678385 , 0.77215382],
 [ 1.01704254 , 0.2355198 ],
 [-0.58026139 ,-0.61917361],
 [ 0.7980995  ,-0.56456529]])


data*=10

## Four point data
# data = np.array([[ 0.9729797, -0.40156498],
#  [ -1.10507727, -0.25586236],
#  [ 0.90864922, 0.55327502],
#  [-0.27104698,  0.93909997]])

fig = plt.figure(figsize=(6,6))
board = plt.axes(xlim=(-20, 20), ylim=(-20, 20), aspect=1)
# radii = np.linspace(0, 12, 6)
radii = np.array([0, 2, 3, 7, 9.5, 12])

connection_dict = {}
for i in range(len(data)):
    connection_dict[i] = []

for r in radii:
    # board.set_title("Rips Filtration")
    board.set_axis_off()
    board.set(xlim=(-20, 20), ylim=(-20, 20), aspect=1)
    # board.annotate('Radius='+str('%.3g'% r ),
                # xy=(20,20), xycoords='figure points')
    for i in range(len(data)):
        pt = data[i]
        circle = plt.Circle((pt[0], pt[1]), r, fc='#0000ffff', ec='#0000ffff', alpha=0.2)
        center = plt.Circle((pt[0], pt[1]), 0.3, fc='k', ec='k')
        center.set_zorder(10.0)
        board.add_patch(circle)
        board.add_patch(center)
    for i in range(len(data)):
        for j in range(len(data)):
            pt1 = data[i]
            pt2 = data[j]
            if i != j and euclidean(pt1, pt2) < 2*r:
                if j != i:
                    if j not in connection_dict[i]:
                        connection_dict[i].append(j)
                    if i not in connection_dict[j]:
                        connection_dict[j].append(i)
                intersect = np.intersect1d(connection_dict[i], connection_dict[j])
                if len(intersect) > 0:
                    for third_pt in intersect:
                        poly = Polygon([data[i], data[j], data[third_pt]],
                                       facecolor='#ccccccff', alpha=0.5)
                        poly.set_capstyle('round')
                        plt.gca().add_patch(poly)
                connectpoints(pt1, pt2)
    plt.draw()
    if r == 0:
        plt.pause(1)
    plt.pause(0.05)
    plt.savefig(f'C:\\Users\\amish\OneDrive\\Documents\\FAU\\Research\\Conferences\\Data Science FAU\\rips_example\\rips_example_r_{r}.svg')
    board.cla()
plt.show()


# diagrams = ripser(data)['dgms']
# print(diagrams)
# plot_diagrams(diagrams, show=True, title="Rips Persistence Diagram")
rips = cm.Rips(maxdim=1) #Go up to 1D homology
rips.build(data/2)
dgmsrips = rips.diagrams()
fig, ax = plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Birth', fontsize=16)
plt.ylabel('Death', fontsize=16)
plot_diagrams(dgmsrips, show=False, ax=ax, size=40)
plt.legend(fontsize=16, loc='lower right')
plt.show()