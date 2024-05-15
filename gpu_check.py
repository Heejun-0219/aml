import tensorflow as tf

# TensorFlow가 사용할 수 있는 GPU 장치를 출력
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
