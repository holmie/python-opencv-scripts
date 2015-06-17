#!/usr/bin/env python
"""
OpenCV detect people and faces
written by Espen Holm Nilsen in about 35 minutes on a Tuesday.
When positive detection archives: detected objects into separate images, full original frame
and the original frame with objects detected outlined into boxes. (blue box; person, green box; face)"""

import cv2, sys, time, json
from copy import deepcopy
from copy import copy

# Where the xml files are kept
cascRoot = '/root/facedet/xml/'
# these files should be in that directory:
cascPath = cascRoot + """haarcascade_frontalface_alt2.xml"""
cascBody = cascRoot + """haarcascade_fullbody.xml"""

# open video capture device (id)
vcap = cv2.VideoCapture(1)
faceCascade = cv2.CascadeClassifier(cascPath)
bodyCascade = cv2.CascadeClassifier(cascBody)

i = 0

# write a status file to status_file path with current number of persons and faces detected, I use this for zabbix monitoring and graphing
status_file = '/tmp/cv.status.json'
# path to archive directory
img_archive = '/var/www/archive/'

def set_res(cap, x, y):
	cap.set(3, int(x))
	cap.set(4, int(y))
	return str(cap.get(3)),str(cap.get(4))

# set resolution 960x720, might be different for your video input device.
set_res(vcap, 960, 720)

num_faces = 0
num_bodies = 0

i = 0

while True:
	ret, frame = vcap.read()
	origframe = copy(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)

	bodies = bodyCascade.detectMultiScale(
		gray,
		scaleFactor=1.05,
		minNeighbors=3,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)

	print "iterate.."

	if len(faces) > 0:
		print """%d faces detected.""" % (len(faces))

	faces_img = []
	bodies_img = []

	for (x, y, w, h) in faces:
		crop_img = frame[y:(y+h), x:(x+w)]
		faces_img.append(crop_img)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	disregard = []
	# I tried to eliminate some false positives by defining these arrays:
	# format is: 
	# two first numbers: <location of border from>, <location of border to>
	# two second numbers: <width of object, from>, <width of object, to>
	# so an object with an X point of 125 - 155, being 310 - 400 in width:
#	disregard.append([125,155, 310, 400])
	# you can print the image.shape in the object iteration to debug these values

	for (x, y, w, h) in bodies:

		crop_img = frame[y:(y+h), x:(x+w)]

		disregarding = 0
		for dr in disregard:
			if y > dr[0] and y < dr[1] and crop_img.shape[0] > dr[2] and crop_img.shape[0] < dr[3]:
				print """Odd body, disregarding"""
				disregarding = 1
#		if y > 28 and y < 45 and crop_img.shape[0] > 95 and crop_img.shape[0] < 150:
#			print """Seems like odd body... disregarding"""
#			continue
		if disregarding == 1:
			continue
		else:
			bodies_img.append(crop_img)
			# in the disregard array you use these values:
			print y
			print crop_img.shape
			
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

	ts = int(time.time())

	# actually write the object images if found:
	for idx, img in enumerate(faces_img):
		fn = """%s%d_face_%d.jpg""" % (img_archive, ts, idx)
		print img
		cv2.imwrite(fn, faces_img[idx])

	for idx, img in enumerate(bodies_img):
		fn = """%s%d_body_%d.jpg""" % (img_archive, ts, idx)
		cv2.imwrite(fn, bodies_img[idx])

	# if objects found, write the frame and frameorig as well
	if len(bodies_img) > 0 or len(faces_img) > 0:
		cv2.imwrite("""%s%d_frame.jpg""" % (img_archive, ts), frame)
		cv2.imwrite("""%s%d_frameorig.jpg""" % (img_archive, ts), origframe)

	fp = open(status_file, 'w')
	fp.write(json.dumps({'bodies': len(bodies_img), 'faces': len(faces_img)}))
	fp.close()

	i += 1

	if i == 5:
		# you can write every 5th frame to a png to get the actual "webcam image", use origframe to avoid seeing any object detection
#		cv2.imwrite("""/var/www/current.png""", frame)
		i = 0

