from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import cv2

cap = cv2.VideoCapture(0)
model = VGG16(include_top=True, weights='imagenet')

wait_period = 100 # ms
while(True):
    ret, frame = cap.read()

    image = img_to_array(frame)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    prediction = list()
    for x, y, z in label[0]:
        prediction.append([y, z])
    print("predictions: ")
    for i in range(5):
        print(i+1,'.',prediction[i])

    cv2.imshow('frame',frame)
    if cv2.waitKey(wait_period) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
