
# coding: utf-8

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.models import load_model

a = np.random.rand(64,64,3) * 255
im_out = Image.fromarray(a.astype('uint8')).convert('RGB')

fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        a = np.random.rand(64,64,3) * 255
        axs[i,j].imshow(Image.fromarray(a.astype('uint8')).convert('RGB'))
        axs[i,j].axis('off')
        cnt += 1

fig.savefig("Random_output.png")

os.chdir("..")
cnn = load_model("final_models/discriminator_90000.h5")
a = np.random.rand(64,64,3)
image = np.array([np.array(Image.fromarray(a.astype('uint8')).convert('RGB'))])
cnn.predict(image)