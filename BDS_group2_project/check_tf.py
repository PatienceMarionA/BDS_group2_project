import tensorflow as tf
print(f'TensorFlow {tf.__version__} - Ready!')
print(f'GPU Available: {len(tf.config.list_physical_devices("GPU")) > 0}')

