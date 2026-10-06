import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt 

training_data_folder = r"D:\AI\CV\Project\Face recognition with openCV\temp_repo\Chapter07\Exercise7.01\dataset\training-data"
test_data_folder = r"D:\AI\CV\Project\Face recognition with openCV\temp_repo\Chapter07\Exercise7.01\dataset\test-data"

# Read a sample image
image_path = R"D:\AI\CV\Project\Face recognition with openCV\temp_repo\Chapter07\Exercise7.01\dataset\test-data\0\Junichiro_Koizumi_0021.jpg"
image = cv2.imread(image_path)
# Showing the image 
cv2.imshow("Image from category", image)
cv2.waitKey(0)

haarcascade_frontalface = r"D:\AI\CV\Project\Face recognition with openCV\haarcascade_frontalface.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)

def detect_face(input_img):
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray)

    if (len(faces)==0):
        return -1, -1

    (x, y, w, h) = faces[0]

    return img_gray[y:y+h, x:x+w], faces[0]

def prepare_data(training_data_folder):
    detected_faces = []
    labels = []

    training_image_dir_name = os.listdir(training_data_folder)

    for dir_name in training_image_dir_name:
        label = int(dir_name)

        training_image_dir_path = os.path.join(training_data_folder, dir_name)
        for image_name in os.listdir(training_image_dir_path):
            image_path = os.path.join(training_image_dir_path, image_name)
            image = cv2.imread(image_path)

            face, rect = detect_face(image)
            if face is not None:
                resized_face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                labels.append(label)
    return detected_faces, labels

detected_faces, labels = prepare_data(training_data_folder)
print(f"Detected faces: {len(detected_faces)}")
print(f"Labels: {len(labels)}")


# Eigen face
# Recognize the face identity
eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()
eigenfaces_recognizer.train(detected_faces, np.array(labels))

def draw_rectangle(test_img, rect):
    x, y, w, h = rect
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def write_text(test_img, label_text, x, y):
    cv2.putText(test_img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

tags = ['0', '1', '2', '3', '4']

def predict(test_img):
    detected_face, rect = detect_face(test_img)
    resized_test_img = cv2.resize(detected_face, (120, 120), interpolation=cv2.INTER_AREA)
    label = eigenfaces_recognizer.predict(resized_test_img)
    label_text = tags[label[0]]
    draw_rectangle(test_img, rect)
    write_text(test_img, label_text, rect[0], rect[1]-5)
    return test_img, label_text

test_img_path = r"D:\AI\CV\Project\Face recognition with openCV\temp_repo\Chapter07\Exercise7.01\dataset\test-data\6\Ana_Guevara_0006.jpg"
test_img = cv2.imread(test_img_path)
predicted_image, label = predict(test_img) 


fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title('actual class: ' + tags[1]+ ' | ' + 'predicted class: ' + label)
plt.axis("off")
plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
plt.show()

