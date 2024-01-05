from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocess
import scipy

def test_model(model):
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    test_generator = test_datagen.flow_from_directory(
        'C:/Users/kart8/PycharmProjects/classifybird/testData',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print("\nTest accuracy:", test_acc)


