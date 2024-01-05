import Buildmodel
import preprocess
import trainmodel
import savemodel
import testmodel
import predictmodel

# build the tensorflow model
num_classes = 2  # Bird and non-bird
model = Buildmodel.build_model(num_classes)
model.summary()

#pre-process the data
train_generator=preprocess.preprocess()

#train the model
trainedmodel=trainmodel.train_model(model,train_generator)

#test model
testmodel.test_model(trainedmodel)
# Check user input and execute the function accordingly

#save model
#download the model
# Get user input
user_input = input("Do you want to save the model? (y/n): ")
if user_input.lower() == 'y':
    savemodel.save_model(trainedmodel)
    print("Model saved!")

#predict model
image_path = 'C:/Users/kart8/PycharmProjects/classifybird/testData/bird/starling.jpg'
#image_path = 'C:/Users/kart8/PycharmProjects/classifybird/testData/nonbird/car.jpg'
predictmodel.predict(trainedmodel,image_path)
