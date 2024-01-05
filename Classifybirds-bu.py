import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocess
import scipy
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model


def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


num_classes = 2  # Bird and non-bird
model = build_model(num_classes)
model.summary()


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/kart8/PycharmProjects/classifybird/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=len(train_generator),
    verbose=4
)
model.save('C:/Users/kart8/PycharmProjects/classifybird/my_model.h5')

# Load the saved Keras model
loaded_model = load_model('C:/Users/kart8/PycharmProjects/classifybird/my_model.h5')

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('C:/Users/kart8/PycharmProjects/classifybird/my_model.tflite', 'wb') as f:
    f.write(tflite_model)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/kart8/PycharmProjects/classifybird/testData',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nTest accuracy:", test_acc)



def predict_bird(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    bird_probability = prediction[0][0]

    if bird_probability > 0.5:
        return "Bird", bird_probability
    else:
        return "Not Bird", bird_probability


#image_path = 'C:/Users/kart8/PycharmProjects/classifybird/testData/bird/starling.jpg'
image_path = 'C:/Users/kart8/PycharmProjects/classifybird/testData/nonbird/car.jpg'
prediction,bird_probability = predict_bird(image_path)
print("Prediction:", image_path,prediction, bird_probability)