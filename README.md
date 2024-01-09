# Bird-Identifier

This project generates a custom model in two ways. 

Method One:
Custom model generation for bird identification of Bluejays, Cardinals, Doves, Sparrows, and Starlings with computer vision and tensorflow.

1. Uses a training dataset consisting of bird photos in Data/dataset/bird and non-bird. It uses tensorFlow with MobileNetV2 architecture.
2. The model is saved as birdmodel.h5 and birdmodel.tflite which can be used as a tflite model to detect birds.


Method Two:
1. Using the TensorFlow's Model  generator Colab notebook.
2. Use bird jpg files in Data/train annotated with labelimg.
3. Download the download the model from the Colab notebook and use it in your projects. 

References:
1. https://medium.com/@kkapa726/exploring-bird-detection-with-tensorflow-a-step-by-step-guide-3f71277419d9
2. https://www.youtube.com/watch?v=-ZyFYniGUsw&t=542s
3. https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb 
4. https://pypi.org/project/labelImg/ 

Notes:
1. Use python virtual enviornment with version 3.9 for TensorFlow