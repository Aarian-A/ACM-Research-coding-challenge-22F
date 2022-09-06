import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('../input/carsforsale/cars_raw.csv')

#First graph displaying original data
dataset['Price'] = dataset["Price"].str.replace('$' , '')
dataset['Price'] = dataset["Price"].str.replace(',' , '')
dataset.drop(dataset[dataset.Price.str.isnumeric()== False].index, inplace = True)

dataset['Price'] = pd.to_numeric(dataset['Price'],errors='coerce')

x = dataset["ValueForMoneyRating"].values
y = dataset["SellerRating"].values
z = dataset["ReliabilityRating"].values

bar = plt.figure(figsize =(16, 9))
ax = plt.axes(projection = "3d")
my_cmap = plt.get_cmap('summer')

ax.set_xlabel("Value For Money Rating")
ax.set_ylabel("Seller Rating")
ax.set_zlabel("Reliability Rating")

threesurf = ax.plot_trisurf(x, y, z, cmap = my_cmap,
                         linewidth = 0.2,
                         antialiased = True,
                         edgecolor = 'grey') 

bar.colorbar(threesurf, ax = ax, shrink = 0.5, aspect = 5)
ax.view_init(35, 235)
plt.show()

#Second graph displaying a 3D scatterplot
x_pred = np.linspace(0, 5, 50)   
y_pred = np.linspace(0, 5, 50)  
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
#model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

plt.style.use('default')
fig = plt.figure(figsize=(25, 25))
ax1 = fig.gca(projection='3d')

ax1.plot(x, y, z, color='g', zorder=15, linestyle='none', marker='o', alpha=0.5)

Z=0.00476584*X + 0.82046486*Y + 0.6748162321603872
ax1.plot_surface(xx_pred, yy_pred, Z)

ax1.set_xlabel('Value for Money Rating', fontsize=12)
ax1.set_ylabel('Seller Rating', fontsize=12)
ax1.set_zlabel('Reliability Rating', fontsize=12)
ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')
plt.show()

#third graph showing 2d line of best fit w/ 2d scatterplot
reg = linear_model.LinearRegression()
reg.fit(dataset[['SellerRating', 'ReliabilityRating']], dataset.ValueForMoneyRating)

X = dataset['ValueForMoneyRating'].values[:,np.newaxis]


Y = dataset['SellerRating'].values
Z = dataset['ReliabilityRating'].values

model2 = linear_model.LinearRegression()
model2.fit(X, Y, Z)
fig2 = plt.figure(figsize=(7, 7))

plt.scatter(X, Y, Z, color='g')
plt.plot(X, model2.predict(X),color='k')

plt.xlabel("Value for Money Rating")
plt.ylabel("Seller Rating")
#plt.zlabel("Reliability Rating")

plt.show()
