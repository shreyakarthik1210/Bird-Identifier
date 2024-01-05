import numpy as np
from tensorflow.keras.preprocessing import image
import Buildmodel
import preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocess
import scipy
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2

def train_model(model,train_generator):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=30,
        steps_per_epoch=len(train_generator),
        verbose=4
    )
    return model