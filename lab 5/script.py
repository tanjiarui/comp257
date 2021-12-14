import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping

face, target = fetch_olivetti_faces(data_home='./', return_X_y=True)
print(face[0, 0].dtype, target.dtype)
print(face.shape, target.shape)

# feature reduction
pca = IncrementalPCA(n_components=240)
face_reduced = pca.fit_transform(face)
print('explained variance ratio in sum:')
print(sum(pca.explained_variance_ratio_))  # .9870387663977453

# clustering
gmm = GaussianMixture(n_components=140, covariance_type='spherical').fit(face_reduced)
new_face, new_target = gmm.sample(1000)

# modeling
def build_model(unit1, unit2, unit3):
	input_layer = tf.keras.Input((new_face.shape[1],))
	hidden_layer1 = tf.keras.layers.Dense(unit1, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())(input_layer)
	hidden_layer2 = tf.keras.layers.Dense(unit2, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())(hidden_layer1)
	hidden_layer3 = tf.keras.layers.Dense(unit3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())(hidden_layer2)
	output_layer = tf.keras.layers.Dense(new_face.shape[1], activation='relu')(hidden_layer3)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.compile(loss='mse', optimizer='Adamax', metrics=['mse'])
	return model

model = KerasRegressor(build_model)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
param_grid = {'unit1': [100, 150, 200], 'unit2': [50, 75, 100], 'unit3': [100, 150, 200]}
grid_search = GridSearchCV(model, param_grid, n_jobs=2)
grid_search.fit(new_face, new_face, validation_split=.2, epochs=100, callbacks=[early_stopping])
print('best parameter:')
print(grid_search.best_params_)  # {'unit1': 100, 'unit2': 50, 'unit3': 100}
print('best score:')
print(grid_search.best_score_)  # 0.3695308923721313
model = build_model(100, 50, 100)
model.fit(new_face, new_face, validation_split=.2, epochs=100, callbacks=[early_stopping])
model.save('best model')

# evaluation
model = tf.keras.models.load_model('best model')
origin = new_face[2]
recover = model.predict(origin.reshape(1, 240))
origin, recover = pca.inverse_transform(origin).reshape(64, 64), pca.inverse_transform(recover).reshape(64, 64)
plt.axis('off')
plt.imshow(origin)
plt.show()
plt.axis('off')
plt.imshow(recover)
plt.show()