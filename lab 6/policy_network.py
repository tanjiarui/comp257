import numpy as np, tensorflow as tf

class PolicyGradient:
	def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.ep_obs, self.ep_as, self.ep_rs = list(), list(), list()
		self.model = self._build_net()

	def loss(self, y_true, y_pred):
		neg_log_prob = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
		loss = tf.reduce_mean(neg_log_prob * self._discount_and_norm_rewards())  # reward guided loss
		return loss

	def _build_net(self):
		input_layer = tf.keras.Input(shape=self.n_features, name="observations")
		# fc1
		layer = tf.keras.layers.Dense(units=10, activation='relu')(input_layer)
		# fc2
		prob = tf.keras.layers.Dense(units=self.n_actions, activation='softmax')(layer)
		model = tf.keras.models.Model(inputs=input_layer, outputs=prob)
		model.compile(optimizer='Adamax', loss=self.loss)
		return model

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def _discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs

	def learn(self):
		discounted_ep_rs_norm = self._discount_and_norm_rewards()
		# one hot encoding
		# y = tf.keras.utils.to_categorical(np.array(self.ep_as), self.n_actions)
		self.model.fit(np.vstack(self.ep_obs), np.array(self.ep_as), batch_size=np.array(self.ep_as).shape[0])
		self.ep_obs.clear(), self.ep_as.clear(), self.ep_rs.clear()  # empty episode data
		return discounted_ep_rs_norm

	def choose_action(self, observation):
		prob_weights = self.model.predict(observation[np.newaxis, :])  # 所有 action 的概率
		action = np.random.choice([0, 1, 2, 3], p=prob_weights.ravel())  # 根据概率来选 action
		return action