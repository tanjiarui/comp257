import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA

# load data
mnist = fetch_openml('mnist_784')  # might return as a numpy array. if so, please manually convert to a dataframe
print(mnist.keys())
X, Y = mnist['data'], mnist['target']
print(X.loc[0, 'pixel1'].dtype, Y.dtype)
print(X.shape, Y.shape)

# pca decomposition
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print('explained variance ratio:')
print(pca.explained_variance_ratio_)  # 0.09746116 0.07155445
holder = [0 for i in range(X2D.shape[0])]
plt.title('the projection of first component')
plt.scatter(X2D[:, 0], holder)
plt.show()
plt.title('the projection of second component')
plt.scatter(X2D[:, 1], holder)
plt.show()

# incremental pca
pca = IncrementalPCA(n_components=154)
for batch in np.array_split(X, 100):
	pca.partial_fit(batch)
X_reduced = pca.transform(X)
X_recovered = pd.DataFrame(pca.inverse_transform(X_reduced))

def plot_digit(row):
	origin, compress = row[:784], row[784:]
	digit = origin.values.reshape(28, 28)
	plt.axis('off')
	plt.imshow(digit)
	plt.show()
	digit = compress.values.reshape(28, 28)
	plt.axis('off')
	plt.imshow(digit)
	plt.show()
# plot data
digits = pd.concat([X, X_recovered], axis=1)
digits.apply(plot_digit, axis=1)