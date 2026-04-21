import tensorflow as tf

model = tf.keras.models.load_model("AI\\Models\\autoencoder\\autoencoder_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("AI\\Models\\autoencoder\\autoencoder_model.tflite", "wb") as f:
    f.write(tflite_model)