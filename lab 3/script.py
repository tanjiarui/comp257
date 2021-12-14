import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

face, target = fetch_olivetti_faces(data_home='./', return_X_y=True)
print(face[0, 0].dtype, target.dtype)
print(face.shape, target.shape)

# search the candidate k with euclidean
silhouette_avg_list = []
search_space = range(50, 200, 10)
for cluster in search_space:
	agglo = AgglomerativeClustering(n_clusters=cluster)
	cluster_labels = agglo.fit_predict(face)
	# The silhouette_score gives the average value for all the samples.
	silhouette_avg = silhouette_score(face, cluster_labels)
	print('for n_clusters =', cluster, 'The average silhouette_score is:', silhouette_avg)
	silhouette_avg_list.append(silhouette_avg)
plt.plot(search_space, silhouette_avg_list)
plt.xlabel('cluster')
plt.ylabel('silhouette score')
plt.show()

# search the candidate k with manhattan
silhouette_avg_list = []
search_space = range(50, 200, 10)
for cluster in search_space:
	agglo = AgglomerativeClustering(n_clusters=cluster, affinity='manhattan', linkage='average')
	cluster_labels = agglo.fit_predict(face)
	# The silhouette_score gives the average value for all the samples.
	silhouette_avg = silhouette_score(face, cluster_labels)
	print('for n_clusters =', cluster, 'The average silhouette_score is:', silhouette_avg)
	silhouette_avg_list.append(silhouette_avg)
plt.plot(search_space, silhouette_avg_list)
plt.xlabel('cluster')
plt.ylabel('silhouette score')
plt.show()

# search the candidate k with cosine
silhouette_avg_list = []
search_space = range(50, 400, 10)
for cluster in search_space:
	agglo = AgglomerativeClustering(n_clusters=cluster, affinity='cosine', linkage='average')
	cluster_labels = agglo.fit_predict(face)
	# The silhouette_score gives the average value for all the samples.
	silhouette_avg = silhouette_score(face, cluster_labels)
	print('for n_clusters =', cluster, 'The average silhouette_score is:', silhouette_avg)
	silhouette_avg_list.append(silhouette_avg)
plt.plot(search_space, silhouette_avg_list)
plt.xlabel('cluster')
plt.ylabel('silhouette score')
plt.show()

def modeling(cluster, affinity):
	if affinity != 'euclidean':
		agglo = FeatureAgglomeration(n_clusters=cluster, affinity=affinity, linkage='average')
	else:
		agglo = FeatureAgglomeration(n_clusters=cluster, affinity=affinity)
	face_reduced = agglo.fit_transform(face)
	# stratified sampling
	x_train, x_test, y_train, y_test = train_test_split(face_reduced, target, stratify=target, test_size=.2)
	plt.subplot(1, 2, 1)
	plt.hist(y_train)
	plt.title('training set distribution')
	plt.xlabel('class')
	plt.ylabel('number')
	plt.subplot(1, 2, 2)
	plt.hist(y_test)
	plt.title('test set distribution')
	plt.xlabel('class')
	plt.ylabel('number')
	plt.show()
	# classification
	model = LogisticRegression(solver='saga')
	param_grid = {'max_iter': [200, 400, 600, 1000]}
	grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
	grid_search.fit(x_train, y_train)
	print('best estimator:')
	print(grid_search.best_estimator_)
	print('best score:')
	print(grid_search.best_score_)
	predict = grid_search.predict(x_test)
	print(classification_report(y_test, predict))

# modeling
modeling(120, 'euclidean')
modeling(140, 'manhattan')
modeling(190, 'cosine')