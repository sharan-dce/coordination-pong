import tensorflow as tf
from lstm import GRU
class Agent:
	def __init__(self, output_q_values_dims):

		self.output_q_values_dims = output_q_values_dims
		#Construct fully connected layers here and add trainable variables
		self.layers = []
		self.layers.append(tf.keras.layers.Conv2D(8, 5, 2, 'valid', activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Conv2D(8, 5, 2, 'valid', activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Reshape((-1,)))
		self.layers.append(tf.keras.layers.Dense(64, activation = tf.nn.relu))
		
		self.layers.append(GRU(16))
		self.layers.append(GRU(16))
		self.layers.append(GRU(8))

		self.gru_layers = 3

		self.q_values = tf.keras.layers.Dense(output_q_values_dims)

		self.FLAG = True

	def forward_prop(self, view_input):	#returns q values
		output_tensor = view_input
		for layer in self.layers:
			output_tensor = layer(output_tensor)
		if self.FLAG:
			self.FLAG = False
			self.trainable_variables = self.q_values.trainable_variables

			for layer in self.layers:
				self.trainable_variables += layer.trainable_variables

		return self.q_values(output_tensor)

	def get_weights(self):
		return_value = []
		for layer in self.layers + [self.q_values]:
			return_value.append(layer.get_weights())
		return return_value

	def set_weights(self, weights):
		layer_list = self.layers + [self.q_values]
		for i, parameter in enumerate(weights):
			layer_list[i].set_weights(parameter)

	def reset(self):
		for layer in self.layers[-self.gru_layers: ]:
			layer.reset()