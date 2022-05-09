import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

##==================================
# PRETRAINED KERAS MODELS
#===================================
# KERAS HAS 4 PRETRAINED IMAGE MODELS
# VGG MOST POPULAR BUT USES A LOT OF MEMOERY.
# RESNET50 BY MICROSOFTIS BEST
# XCEPTION IS ALSO GOOD
# INCEPTION V3 BY GOOGLE IS NOT AS POWERFUL
##=====================================
#=================================

##==================================
# IMPORT MPRETRAINED RESNET IMAGE MODEL
#===================================
# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model)
# SIZE IMAGE MUST MATCH NODES.
# RESNET IS 224 X 224
img = image.load_img("bay.jpg", target_size=(224, 224))


##==================================
# CONVERT TO PLAIN NUMBERS
#===================================
# CONVERT TO 3 DIMENTION ARRAY
# Convert the image to a numpy array
# DIMENSION X X H. COLOUR
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
# SO ADD 4TH DIMENTION
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
# EPAND DIM TO NORMALISE DATA TO MAKE IT SMALLER
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
# NOW RUN NORMALISED DATA TO RETURN THE PREDICTION OBJECT
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
#1000 ARRAY OF FLOATING POINT NUMBERS, WILL CHECK 100 UMBERS TO SEE IF THEY ARE WHAT THEY ARE LOOKING FOR
# GIVES TOP 9 MOST MATCHED NODES PREDICTION
# NEED INTERENT PROCESS TO DOWNLOAD
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

