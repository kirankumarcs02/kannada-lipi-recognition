import cv2
from keras.models import load_model
import numpy as np

model1 = load_model('kannada_cnn_model.h5')
b,g,r,a = 0,255,0,0
letter= {
    1: "\u0C85",
    2: "\u0C86",
    3: "\u0C87",
    4: "\u0C88",
    0: "",
    '': ''
}

letters= {
    1: "1_a",
    2: "2_aa",
    3: "3_i",
    4: "4_i",
    0: "",
    '': ''
}
def main():
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans = ''
        classes=''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                newImage_thresh = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage_thresh, (48, 48))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(newImage.shape[0], 1)
                ans, classes = keras_predict(model1, newImage)
        print('classes = ', classes)
        print('ans1 = ', ans)
        print(letter[classes])
        x, y, w, h = 0, 0, 300, 300

        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Predicted letter : " + str(letters[classes]), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 48
    image_y = 48
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

# print(model1.summary())
keras_predict(model1, np.zeros((48, 48), dtype=np.uint8))
print(model1)
main()
