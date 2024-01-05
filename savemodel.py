import tensorflow as tf
from keras.models import load_model

def save_model(model):
    model.save('C:/Users/kart8/PycharmProjects/classifybird/birdmodel.h5')

    # Load the saved Keras model
    loaded_model = load_model('C:/Users/kart8/PycharmProjects/classifybird/my_model.h5')

    # Convert the Keras model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('C:/Users/kart8/PycharmProjects/classifybird/birdmodel.tflite', 'wb') as f:
        f.write(tflite_model)
