import settings
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


class DigitClassifier:

	def train_model(self, X_train, y_train):
		start_time = time.time()

		model = keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(512, activation=tf.nn.relu),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10, activation=tf.nn.softmax)
		])
		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

		tensorboard = TensorBoard(log_dir=settings.lOGS_DIR)

		history = model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard])
		end_time = time.time()
		print('Total train time = ', round(end_time - start_time), 's')

		self._visualize_model_training(history)
		return model

	def evaluate_model(self, model, X_test, y_test):
		y_test_predict = model.predict(X_test)
		predicted_label = []
		for y_test_i in y_test_predict:
			label = np.argmax(y_test_i)
			predicted_label.append(label)

		self._calculate_model_metrics(y_test, predicted_label)
		# test_loss, test_acc = model.evaluate(X_test, y_test)
		# print('Test accuracy = ', test_acc)

	@staticmethod
	def _calculate_model_metrics(y, y_pred):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			y, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())

	@staticmethod
	def _visualize_model_training(history):
		print(history.history.keys())
		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('Number of epoch')
		plt.legend(['train'], loc='upper left')
		plt.show()

	@staticmethod
	def save_model(model):
		try:
			keras.models.save_model(model, settings.MODELS_DIR + settings.MODEL_FILE_NAME)
		except IOError:
			raise ValueError('Something wrong with file save operation.')
		except ValueError:
			raise ValueError('Something wrong with model.')

	@staticmethod
	def load_my_model():
		try:
			model = keras.models.load_model(settings.MODELS_DIR + settings.MODEL_FILE_NAME)
			return model
		except IOError:
			raise ValueError('Something wrong with file save operation.')
