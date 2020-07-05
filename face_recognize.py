import face_recognition
import argparse
import pickle
import cv2
 
ap = argparse.ArgumentParser()  # argument parser
ap.add_argument("--encodings", required=True, help="path of the serialized facial encodings")
ap.add_argument("--image", required=True, help="path to input image")
ap.add_argument("--detection-method", type=str, default="hog")  # detection method set to HOG
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())   # loading the previously created encodings for training data.
 
image = cv2.imread(args["image"])    # Image in which we want to detect and recognize the face
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # converting the same image from BGR to RGB
 
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])   # detect the (x, y)-coordinates of the bounding boxes corresponding to each face
encodings = face_recognition.face_encodings(rgb, boxes)    # creating the new encodings for the faces in the image
 
names = []  # initialize the list of names for each face detected

# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known encodings
	matches = face_recognition.compare_faces(data["encodings"], encoding)
	name = "Unknown"   # setting the name to unknown is there is no match
    
	if True in matches:
		# find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
		matchedIds = [i for (i, b) in enumerate(matches) if b]  # it will contain all the ids for True match
		counts = {}
 
		# loop over the matched indexes and maintain a count for each recognized face face
		for i in matchedIds:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
 
		# determine the recognized face with the largest number of votes 
		name = max(counts, key=counts.get)
	
	names.append(name)  # update the list of names
    
# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)  # draw the rectangle on detected face
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)  # putting the text on the recognized person
 
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
