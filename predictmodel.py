import numpy as np
from tensorflow.keras.preprocessing import image
def predict_bird(model,image_path):
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

def predict(model,image_path):
  prediction,bird_probability = predict_bird(model,image_path)
  print("Prediction:", image_path,prediction, bird_probability)