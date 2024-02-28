import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        probability = model.predict(test_image)[0][0]

        if probability > 0.95:
            message = "The patient is completely out of danger as the probability of not having cancer is > 0.95."
        elif probability > 0.5:
            message = "The patient may not have cancer, but further evaluation is recommended."
        else:
            message = "The patient is at risk, and further medical attention is advised."

        return [message]
