import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adamax


# Initializing arrays to store the losses of generator and discriminator at every epoch
D_loss = []
G_loss = []


# Building the class for the GAN
class GAN():

    def build_generator(self, image_dim = (64, 64, 3)):   # Function to build generator, argument - image dimensions to be given to get the same dimensions as the output image

        # Rescaling to consider in the upsampling
        size_width = image_dim[0] // 4
        size_height = image_dim[1] // 4

        # Giving the shape to our noise vector
        noise_shape = (100,)

        # Building model architecture discussed in the report
        model = Sequential()
        model.add(Dense(256 * size_width * size_height, activation="relu", input_shape=noise_shape))
        model.add(Reshape((size_width, size_height, 256)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        # Returning model
        return Model(noise, img)

    def build_discriminator(self, image_dim = (64, 64, 3)):  # Function to build discriminator, argument - image dimensions to be give input dimensions

        # Building model architecture discussed in the report
        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, strides=2, input_shape=image_dim, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=image_dim)
        validity = model(img)

        return Model(img, validity)



    def __init__(self, image_dim = (64, 64, 3), learning_rate = 0.0002):

        self.image_dim = image_dim
        self.lr = learning_rate
        # making an optimizer for training
        optimizer = Adam(self.lr, 0.5)

        # making instance of discriminaor
        self.discriminator = self.build_discriminator(image_dim)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # For the combined model i.e the DCGAN we will only train the generator
        self.discriminator.trainable = False

        # making instance of generator
        self.generator = self.build_generator(image_dim)
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        noise = Input(shape=(100,))

        img = self.generator(noise)
        valid = self.discriminator(img)

        # making instance of combined generator and discriminator hence the GAN
        self.DCGAN = Model(noise, valid)
        self.DCGAN.summary()
        self.DCGAN.compile(loss='binary_crossentropy', optimizer=optimizer)



    def train(self, n, batch_size=128, save_interval=50):

        # Loading the face dataset
       	X = np.load("Face_Dataset_64RGB.npy").astype("float32") / 255
       	print("Shape of Dataset : " )
       	print(X.shape)

        for epoch in range(n):

            # Curating random samples from dataset on which the discriminator is to be trained on.
            generate_random_indices = np.random.randint(0, X.shape[0], batch_size)

            # Selecting images from the generated random indices
            imgs = X[generate_random_indices]

            # Generating noise as input for generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(noise)

            # Training Discriminator on real images
            # Making array of value one's since all the images are real
            y_true = np.ones((batch_size, 1))
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(imgs, y_true)

            # Training Discriminator on Fake images genetated by the generator
            # Making array of value zero's since all the images generated were fake
            y_fake = np.zeros((batch_size, 1))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, y_fake)

            # Taking average of the losses obtained from the real and fake images during training of the discriminator
            d_loss = np.add(d_loss_real, d_loss_fake) / 2

            self.discriminator.trainable = False
            # Generating noise to train our GAN
            noise = np.random.normal(0, 1, (64, 100))

            # Training the DCGAN
            # Making arry of 1 since the output of DCGAN is the output of discriminator given the input  of generator suggesting that
            # the output probability of the discriminator must be 1 so that the images produced by the generator are similiar to the real data 
            y = np.ones((64, 1))
            g_loss = self.DCGAN.train_on_batch(noise, y)

            # Printing out the losses
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # Saving the losses for the discriminator model and the GAN model
            D_loss.append(np.array([epoch, d_loss[0], d_loss[1]]))
            G_loss.append(np.array([epoch, g_loss]))

            # Saving the images, losses and the model as per the epoch and save interval, model is saved every 10,000 epoch
            if epoch % save_interval == 0:
                np.save("D_loss", np.array(D_loss))
                np.save("G_loss", np.array(G_loss))
                noise = np.random.normal(0, 1, (25, 100))
                gen_imgs = self.generator.predict(noise)
                fig, axs = plt.subplots(5, 5)
                cnt = 0
                for i in range(5):
                    for j in range(5):
                        axs[i,j].imshow((gen_imgs[cnt] * 255).astype("uint8"))
                        axs[i,j].axis('off')
                        cnt += 1
                fig.savefig("Output/%d.png" % epoch)
                plt.close()
                if epoch % 10000 == 0:
	                self.generator.save("Models/generator_%d.h5" %(epoch))
	                self.discriminator.save("Models/discriminator_%d.h5" %(epoch))
	                self.DCGAN.save("Models/DCGAN%d.h5" %(epoch))
	            

if __name__ == '__main__':
    # Making instance of the model
    dcgan = GAN(image_dim = (64, 64, 3))
    # Training the model
    dcgan.train(n=100001, batch_size=32, save_interval=200)



#-----------------
# architecture referred from https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# and https://arxiv.org/abs/1511.06434
# class structure referred from https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
#-----------------