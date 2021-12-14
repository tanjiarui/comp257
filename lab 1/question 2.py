import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X, t = make_swiss_roll(n_samples=1000)
plot = plt.figure().add_subplot(projection='3d')
plot.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
plt.show()

pca = KernelPCA(n_components=2)
X2D = pca.fit_transform(X)
plt.scatter(X2D[:, 0], X2D[:, 1], c=t)
plt.show()

pca = KernelPCA(n_components=2, kernel='rbf')
X2D = pca.fit_transform(X)
plt.scatter(X2D[:, 0], X2D[:, 1], c=t)
plt.show()

pca = KernelPCA(n_components=2, kernel='sigmoid')
X2D = pca.fit_transform(X)
plt.scatter(X2D[:, 0], X2D[:, 1], c=t)
plt.show()

t = np.where(t <= t.mean(), 0, 1)
pipeline = Pipeline([('pca', KernelPCA(n_components=2)), ('model', LogisticRegression())])
param_grid = {'pca__kernel': ['linear', 'rbf', 'sigmoid'], 'pca__gamma': np.arange(0, 1, .02), 'model__max_iter': [1000, 2000, 4000, 5000]}
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', refit=True)
grid_search.fit(X, t)
print('best estimator:')
print(grid_search.best_estimator_)
print('best score:')
print(grid_search.best_score_)