import face_recognition
import json
import cv2

# load the stored facial information from the JSON file
with open('faces.json', 'r') as f:
    face_info = json.load(f)

# initialize the camera
cap = cv2.VideoCapture(0)

# capture a new image from the camera
ret, frame = cap.read()

# convert the image from BGR (OpenCV) to RGB (face_recognition)
rgb_frame = frame[:, :, ::-1]

# find all the faces in the new image
new_face_locations = face_recognition.face_locations(rgb_frame)

# get the facial features for each face in the new image
new_face_encodings = face_recognition.face_encodings(rgb_frame, new_face_locations)

flag=1
# loop through each detected face in the new image
for new_face_encoding in new_face_encodings:
    # loop through each stored face encoding and compare to the new face encoding
    for stored_face_encoding in face_info.values():
        # compare the face encodings
        match = face_recognition.compare_faces([stored_face_encoding['encoding']], new_face_encoding)

        # if there's a match, print the time the image was captured
        if match[0]:
            print("Match found for image captured")
            flag=0
            break

if(flag):
    print("not found")

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
