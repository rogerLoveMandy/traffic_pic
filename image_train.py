from  net.lenet import LeNet


import matplotlib
matplotlib.use('Agg')


from  keras.preprocessing.image import   ImageDataGenerator  #数据增强器
from keras.optimizers import   Adam

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import  img_to_array

from  keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as  np
import argparse
import cv2
import  os
import  sys
sys.path.append('..')
from imutils import paths






EPOCHS = 35
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 62
norm_size = 32


def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    np.random.seed(42)
    np.random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels


def train(aug, trainX, trainY, testX, testY):
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save("image_model.h5")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.jpg")




if __name__=='__main__':
    train_file_path = "data/train"
    test_file_path = "data/test"
    trainX,trainY = load_data(train_file_path)
    testX,testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(aug,trainX,trainY,testX,testY)