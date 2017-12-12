import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import normalize

generator_loss = np.load("G_loss.npy")
discrimator_loss = np.load("D_loss.npy")

epoch = []
G_loss_Y = []


for arr in generator_loss:
	epoch.append(arr[0])
	G_loss_Y.append(arr[1])

D_loss_Y = []
D_accuracy = []

for arr in discrimator_loss:
    D_loss_Y.append(arr[1])
    D_accuracy.append(arr[2])



print(np.mean(G_loss_Y))

print(np.mean(D_loss_Y))

print(np.mean(D_accuracy))


plt.rcParams["figure.figsize"] = [16,9]

plt.xlabel('steps / epochs')

plt.ylabel('loss')

plt.title('DCGAN losses')

plt.plot(epoch, G_loss_Y, label = 'generator loss', color = 'red')

plt.plot(epoch, D_loss_Y, label = 'discrimator loss', color = 'blue')

plt.grid(True)

plt.legend()

plt.show()


plt.xlabel('steps / epochs')

plt.ylabel('Discriminator accuracy')

plt.title('DCGAN losses')

plt.plot(epoch, D_accuracy, label = 'Discriminator accuracy', color = '#72f97e')

plt.grid(True)

plt.legend()

plt.show()