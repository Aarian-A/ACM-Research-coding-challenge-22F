import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 5, 50)   
y = np.linspace(0, 5, 50) 

X,Y = np.meshgrid(x,y)
Z=0.00476584*X + 0.82046486*Y + 0.6748162321603872

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z)
