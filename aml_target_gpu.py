import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
import time

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

def generate_targeted_adversaries_batch(model, images, targets, eps=2 / 255.0):
    # Cast the images
    images = tf.cast(images, tf.float32)

    # Record our gradients
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        loss = CategoricalCrossentropy()(targets, preds)

    # Calculate the gradients of loss with respect to the images, then compute the sign of the gradient
    gradients = tape.gradient(loss, images)
    signedGrad = tf.sign(gradients)

    # Construct the image adversaries by subtracting the signed gradient
    adversaries = (images - (signedGrad * eps)).numpy()
    return adversaries

# GPU 사용 가능 여부 확인
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

# GPU 장치에서 모델 학습
with tf.device('/GPU:0'):
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)

# 측정 시작
start_time = time.time()

# 적대적 샘플 배치 생성
batch_size = 64  # 배치 사이즈 설정
num_batches = len(testX) // batch_size
total_time = 0

for i in range(num_batches):
    batch_images = testX[i * batch_size: (i + 1) * batch_size]
    batch_labels = testY[i * batch_size: (i + 1) * batch_size]
    target_labels = (np.argmax(batch_labels, axis=1) + 1) % 10  # Choose target classes different from true classes
    target_labels = to_categorical(target_labels, num_classes=10)

    batch_start_time = time.time()
    adversaries = generate_targeted_adversaries_batch(model, batch_images, target_labels, eps=0.1)
    batch_end_time = time.time()

    total_time += (batch_end_time - batch_start_time)

# 측정 종료
end_time = time.time()
elapsed_time = end_time - start_time

print(f"배치 처리 후 적대적 샘플 생성 시간: {elapsed_time:.4f} 초")
print(f"배치 당 평균 적대적 샘플 생성 시간: {total_time / num_batches:.4f} 초")

# 60,000개 샘플 생성 예상 시간 계산
total_time_per_image = total_time / len(testX)
total_time_60000 = total_time_per_image * 60000
print(f"적대적 샘플 60,000개 생성 예상 시간: {total_time_60000 / 3600:.2f} 시간")

# Visualize and compare the original and adversarial images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    image = testX[i]
    true_label = np.argmax(testY[i])
    target_label = (true_label + 1) % 10  # Choose a target class different from true class
    adversary = generate_targeted_adversaries_batch(model, image.reshape(1, 28, 28, 1), to_categorical([target_label], 10), eps=0.1)
    pred = model.predict(adversary)

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
