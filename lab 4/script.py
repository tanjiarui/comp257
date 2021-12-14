import matplotlib.pyplot as plt, numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture

face, target = fetch_olivetti_faces(data_home='./', return_X_y=True)
print(face[0, 0].dtype, target.dtype)
print(face.shape, target.shape)

# feature reduction
pca = IncrementalPCA(n_components=240)
face_reduced = pca.fit_transform(face)
print('explained variance ratio in sum:')
print(sum(pca.explained_variance_ratio_))  # .9870387663977453

# decide the covariance type
gmm = GaussianMixture(n_components=2, covariance_type='spherical').fit(face_reduced)
labels = gmm.predict(face_reduced)
plt.scatter(face_reduced[:, 0], face_reduced[:, 100], c=labels)
plt.show()

#  search the candidate k
bic_list = []
aic_list = []
search_space = range(1, 240, 10)
for cluster in search_space:
	gmm = GaussianMixture(n_components=cluster, covariance_type='spherical').fit(face_reduced)
	bic_list.append(abs(gmm.bic(face_reduced)))
	aic_list.append(abs(gmm.aic(face_reduced)))
plt.plot(search_space, bic_list, label='BIC')
plt.plot(search_space, aic_list, label='AIC')
plt.xlabel('cluster')
plt.ylabel('information criterion')
plt.legend(loc='best')
plt.show()

# random sampling
gmm = GaussianMixture(n_components=140, covariance_type='spherical').fit(face_reduced)
print('hard clustering')
print(gmm.predict(face_reduced))
print('soft clustering')
print(gmm.predict_proba(face_reduced))
face_recovered = pca.inverse_transform(gmm.sample(2)[0])
origin = face_recovered.reshape(2, 64, 64)
plt.axis('off')
plt.imshow(origin[0])
plt.show()
plt.axis('off')
plt.imshow(origin[1])
plt.show()
rotate = np.array([origin[::-1][0].T, origin[::-1][1].T])
plt.axis('off')
plt.imshow(rotate[0])
plt.show()
plt.axis('off')
plt.imshow(rotate[1])
plt.show()
origin = pca.transform(origin.reshape(2, 4096))
rotate = pca.transform(rotate.reshape(2, 4096))
print('score of origin')
print(gmm.score_samples(origin))  # 41.60026599  4.77964417
print('score of rotation')
print(gmm.score_samples(rotate))  # -309.31724632 -333.40877638