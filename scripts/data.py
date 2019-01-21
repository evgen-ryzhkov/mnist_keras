import tensorflow as tf
mnist = tf.keras.datasets.mnist


class MNISTData:

	@staticmethod
	def get_train_and_test_data():
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train, X_test = X_train / 255, X_test / 255
		return X_train, y_train, X_test, y_test

