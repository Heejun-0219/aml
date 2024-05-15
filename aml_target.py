import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

class SimpleCNN:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

def generate_targeted_adversary(model, image, target, eps=2 / 255.0):
    # Create a target label that is the opposite of the true label
    target_label = to_categorical([target], num_classes=10)

    # Cast the image
    image = tf.cast(image, tf.float32)

    # Record our gradients
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        loss = CategoricalCrossentropy()(target_label, pred)

    # Calculate the gradients of loss with respect to the image, then compute the sign of the gradient
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)

    # Construct the image adversary by subtracting the signed gradient
    adversary = (image - (signedGrad * eps)).numpy()
    return adversary

# load MNIST dataset and scale the pixel values to the range [0, 1]
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize our optimizer and model
opt = Adam(learning_rate=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the simple CNN on MNIST
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)

# generate an image adversary for the current image and make a prediction on the adversary
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    image = testX[i]
    true_label = np.argmax(testY[i])
    target_label = (true_label + 1) % 10  # Choose a target class different from true class
    adversary = generate_targeted_adversary(model, image.reshape(1, 28, 28, 1), target_label, eps=0.1)
    pred = model.predict(adversary)

    # Visualize and compare the original and adversarial images
    adversary = adversary.reshape((28, 28)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image.reshape((28, 28)) * 255
    image = image.astype("uint8")

    image = np.dstack([image] * 3)
    adversary = np.dstack([adversary] * 3)
    image = cv2.resize(image, (96, 96))
    adversary = cv2.resize(adversary, (96, 96))

    imagePred = np.argmax(testY[i])
    adversaryPred = pred[0].argmax()
    color = (0, 255, 0) if imagePred == adversaryPred else (0, 0, 255)

    cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    output = np.hstack([image, adversary])
    cv2.imshow("Targeted FGSM Adversarial Images", output)
    cv2.waitKey(0)
