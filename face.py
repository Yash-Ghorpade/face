import cv2
import face_recognition
import json
import time

# initialize the camera
cap = cv2.VideoCapture(0)

# create a dictionary to store the facial information
face_info = {}

# capture a single frame from the camera
ret, frame = cap.read()

# find all the faces in the image
face_locations = face_recognition.face_locations(frame)

# get the facial features for each face
face_encodings = face_recognition.face_encodings(frame, face_locations)

# loop through each detected face
for j, face_encoding in enumerate(face_encodings):
    # get the location of the face
    top, right, bottom, left = face_locations[j]

    # add the face information to the dictionary
    face_info[str(time.time())] = {
        'top': top,
        'right': right,
        'bottom': bottom,
        'left': left,
        'encoding': face_encoding.tolist()
    }

# convert the dictionary to JSON format
face_json = json.dumps(face_info)

# write the JSON to a file
with open('faces.json', 'w') as f:
    f.write(face_json)

# release the camera
cap.release()
