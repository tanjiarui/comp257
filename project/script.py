import numpy as np, tensorflow as tf
from scipy.io import loadmat
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# load data
data = loadmat('umist cropped.mat')['facedat'][0]
face, target = list(), list()
label = 0
for batch in data:
	for sample in batch.T:
		sample = sample.T.reshape(-1)
		face.append(sample)
		target.append(label)
		del sample
	del batch
	label += 1
face, target = np.array(face), np.array(target)
del data

# feature reduction
pca = IncrementalPCA(n_components=260)
face_reduced = pca.fit_transform(face)
print('explained variance ratio in sum:')
print(sum(pca.explained_variance_ratio_))  # 0.9864809639643983

# clustering
gmm = GaussianMixture(n_components=20, covariance_type='spherical').fit(face_reduced)
new_face, new_target = gmm.sample(1000)
new_target = tf.keras.utils.to_categorical(new_target, label)
# stratified sampling
x_train, x_test, y_train, y_test = train_test_split(new_face, new_target, stratify=new_target, test_size=.2)

# modeling
hidden_units = [64, 64]
def create_deep_and_cross_model():
	input_layer = tf.keras.Input(shape=new_face.shape[1])

	cross = input_layer
	for _ in hidden_units:
		units = input_layer.shape[-1]
		x = tf.keras.layers.Dense(units, kernel_regularizer='l2')(input_layer)
		cross = input_layer * x + cross
	cross = tf.keras.layers.BatchNormalization()(cross)

	deep = input_layer
	for units in hidden_units:
		deep = tf.keras.layers.Dense(units, kernel_regularizer='l2')(deep)
		deep = tf.keras.layers.BatchNormalization()(deep)
		deep = tf.keras.layers.ReLU()(deep)

	merged = tf.keras.layers.concatenate([cross, deep])
	output_layer = tf.keras.layers.Dense(units=label, activation='softmax')(merged)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
	return model

model = create_deep_and_cross_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping])
predict = model.predict(x_test)
predict, y_test = [np.argmax(one_hot) for one_hot in predict], [np.argmax(one_hot) for one_hot in y_test]
print(classification_report(y_test, predict))
'''
			precision    recall  f1-score   support

		0       1.00      1.00      1.00         8
		1       1.00      1.00      1.00         7
		2       0.80      1.00      0.89         4
		3       1.00      1.00      1.00         6
		4       1.00      0.93      0.96        14
		5       1.00      1.00      1.00        17
		6       1.00      1.00      1.00        10
		7       1.00      1.00      1.00        10
		8       1.00      1.00      1.00         8
		9       1.00      1.00      1.00         5
		10       1.00      1.00      1.00        24
		11       1.00      0.92      0.96        12
		12       0.92      1.00      0.96        11
		13       1.00      1.00      1.00         3
		14       1.00      1.00      1.00         4
		15       1.00      1.00      1.00        13
		16       1.00      1.00      1.00        10
		17       1.00      1.00      1.00        16
		18       1.00      1.00      1.00         9
		19       1.00      1.00      1.00         9

accuracy                             0.99        200
macro avg        0.99      0.99      0.99        200
weighted avg     0.99      0.99      0.99        200
'''