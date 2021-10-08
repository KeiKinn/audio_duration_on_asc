import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import numpy as np


def loss_plot(train_loss, val_loss, path, epoch,isSave=1):
	fig, ax = plt.subplots()
	ax.plot(train_loss, label='train_loss')
	ax.plot(val_loss, label='val_loss')
	legend = ax.legend()

	ax.set(xlabel='epoch / times', ylabel='loss')
	ax.grid()

	while os.path.isfile(path+'/loss_{}.png'.format(epoch)):
		epoch += 10000
	fig_path = path+'/loss_{}.png'.format(epoch)

	if isSave:
		fig.savefig(fig_path)
	plt.close()


def accuracy_plot(train_accuracy, val_accuracy, path, epoch, isSave=1):
	fig, ax = plt.subplots()
	ax.plot(train_accuracy, label='train_accuracy')
	ax.plot(val_accuracy, label='val_accuracy')
	legend = ax.legend()
	ax.set(xlabel='epoch / times', ylabel='accuracy')
	ax.grid()
	plt.ylim((0, 100))
	my_y_ticks = np.arange(0, 100, 5)
	plt.yticks(my_y_ticks)

	while os.path.isfile(path+'/accuracy_{}.png'.format(epoch)):
		epoch += 10000000
	fig_path = path+'/accuracy_{}.png'.format(epoch)

	if isSave:
		fig.savefig(fig_path)	
	plt.close()
