

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Argument parser to pass the arguments

a = argparse.ArgumentParser()
a.add_argument("-i", "--input", type=str,
	help="path to optional input video (webcam will be used otherwise)")
a.add_argument("-p", "--prototxt", type=str, required=True,
	help="path to Caffe prototxt file")
a.add_argument("-m", "--model", type=str, required=True,
	help="path to Caffe pre-trained model")
a.add_argument("-c", "--points", type=str, required=True,
	help="path to cluster center points")
a.add_argument("-w", "--width", type=int, default=500,
	help="input width dimension of frame")
args = vars(a.parse_args())

# Here Im checking If the input is webcam or video
cam = not args.get("input", False)

# if a video path was not supplied, grab a reference to the webcam
if cam:
	print("[INFO] starting video stream...")
	v = VideoStream(src=0).start()
	time.sleep(2.0)

# get the reference to video
else:
	print("[INFO] opening video file...")
	v = cv2.VideoCapture(args["input"])

# load the model,cluster points and prototext file

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
p = np.load(args["points"])

# Add 1x1 convolutions cluster center to the model
cls = net.getLayerId("class8_ab")
conv = net.getLayerId("conv8_313_rh")
p = p.transpose().reshape(2, 313, 1, 1)
net.getLayer(cls).blobs = [p.astype("float32")]
net.getLayer(conv).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# loop over frames from the video stream
while True:
	
	frame = v.read()
	frame = frame if cam else frame[1]

	
	if not cam and frame is None:
		break

	# change the frame from BGR to LAB space.
	
	frame = imutils.resize(frame, width=args["width"])
	scaled = frame.astype("float32") / 255.0
	lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

	# Resize the lab frame to 224*224
	resized = cv2.resize(lab, (224, 224))
	L = cv2.split(resized)[0]
	L -= 50

	# Pass the L to channel ,so It predicts AB channel
	net.setInput(cv2.dnn.blobFromImage(L))
	ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

	# Resize it to the original image size
	ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
	L = cv2.split(lab)[0]
	colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

	# Convert the image again from LAB to RGB
	colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
	colorized = np.clip(colorized, 0, 1)
	colorized = (255 * colorized).astype("uint8")

	# show the original and final colorized frames
	cv2.imshow("Original", frame)
	cv2.imshow("Grayscale", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	cv2.imshow("Colorized", colorized)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# If it is a cam ,stop the video stream
if cam:
	v.stop()

# Release the pointer
else:
	v.release()

cv2.destroyAllWindows()