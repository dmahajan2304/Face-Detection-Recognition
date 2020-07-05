import cv2
import os
import face_recognition
import argparse
import pickle
from imutils import paths

ap = argparse.ArgumentParser()  # Initializing the argument parser
ap.add_argument("--dataset", required=True, help="path to input directory")
ap.add_argument("--encodings", required=True, help="path of the serialized facial encodings")
ap.add_argument("--detection-method", type=str, default="hog")   # detection method set to HOG
args = vars(ap.parse_args())   # to create dictionary of arguments passed

imagePaths = list(paths.list_images(args["dataset"]))  # we will give the path during the runtime

learntEncodings = []
learntNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]   # extracting name from the directory.
	image = cv2.imread(imagePath)  # reading the image
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting the image from BGR to RGB
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])    # detect the (x, y)-coordinates of the bounding boxes corresponding to each face
	encodings = face_recognition.face_encodings(rgb, boxes)   # compute the encoding for the face 
	for encoding in encodings:
		# add each encoding + name to our set of known names and encoding
		learntEncodings.append(encoding)
		learntNames.append(name)
        
# writing the serialized encodings to pickle file
print("[INFO] serializing encodings...")
data = {"encodings": learntEncodings, "names": learntNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()




