# USAGE
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/massachusetts.mp4 --output output/massachusetts_output.avi
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4 --output output/toronto_output.avi

# import the necessary packages
import numpy as np
from flask import Response
from flask import Flask, jsonify
from flask import render_template
from flask_session import Session
from flask_cors import CORS
import threading
import argparse
import imutils
import time
import cv2
from enet.models.ENet import ENet
import torch
import yaml
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
torch.backends.cudnn.benchmark = True

app = Flask(__name__)
SESSION_TYPE = "filesystem"
Session(app)

fps = 0
total_time = 0
lock = threading.Lock()



CORS(app, resources={r'/*': {'origins': '*'}})


def decode_segmap(temp, plot=False):
        
		colors = [   #[  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    	]
		label_colours = dict(zip(range(19), colors))
		r = temp.copy()
		g = temp.copy()
		b = temp.copy()
		for l in range(0, 2):
		    r[temp == l] = label_colours[l][0]
		    g[temp == l] = label_colours[l][1]
		    b[temp == l] = label_colours[l][2]
		
		rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
		rgb[:, :, 0] = r 
		rgb[:, :, 1] = g 
		rgb[:, :, 2] = b 
		return rgb
# construct the argument parse and parse the arguments

# load the class label names
MODEL_PATH = "./enet-cityscapes/hardnet_cityscapes_best_model.pkl"
OUTPUT_PATH = "~/client/src/assets/test3_ouput.mp4"
VIDEO_PATH = "./videos/test3.mp4"
CFG = "./configs/hardnet.yml"
SHOW = 1
WIDTH = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if a colors file was supplied, load it from disk


# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNet(args["model"])

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(VIDEO_PATH)
writer = None

# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
# enet = ENet(2)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# enet = enet.to(device)
# state_dict = torch.load("./enet-cityscapes/ckpt-enet-1.pth", map_location=torch.device("cpu"))['state_dict']
# enet.load_state_dict(state_dict)
with open(CFG) as p:
	cfg = yaml.load(p)

#print(cfg)

model = get_model(cfg['model'], 19).to(device)
state = convert_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"))['model_state'])
model.load_state_dict(state)
# mean = np.array([104.00699, 116.66877, 122.67892])
model.v2_transform(trt=False) 
model.eval()
model.to(device)

def segment():
	global vs, lock, fps, total_time, writer
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# construct a blob from the frame and perform a forward pass
		# using the segmentation model
		start = time.time()
		frame = cv2.resize(frame, (960,540))
		clone = frame.copy()
		frame = frame.astype(np.float16)

		value_scale = 255
		mean = [0.406,0.456,0.485]
		std = [0.225,0.224,0.229]
		std = [item*value_scale for item in std]
		frame = (frame - mean) /std
		frame = frame.transpose(2,0,1)
		frame = np.expand_dims(frame, 0)
		frame = torch.from_numpy(frame).float()
		frame = frame.to(device)

		with torch.no_grad():
			output = model(frame)
		# blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
		# 	swapRB=True, crop=False)
		# net.setInput(blob)
		pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
		#print(pred)


		
		
		end = time.time()
		mask = decode_segmap(temp=pred)
		#print(mask)



		# infer the total number of classes along with the spatial
		# dimensions of the mask image via the shape of the output array
		#(numClasses, height, width) = output.shape

		# our output class ID map will be num_classes x height x width in
		# size, so we take the argmax to find the class label with the
		# largest probability for each and every (x, y)-coordinate in the
		# image
		#classMap = np.argmax(output, axis=0)

		# given the class ID map, we can map each of the class IDs to its
		# corresponding color
		
		#print(mask.shape)
		#print(clone.shape)
		# resize the mask such that its dimensions match the original size
		# of the input frame
		#mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
		#	interpolation=cv2.INTER_NEAREST)

		# perform a weighted combination of the input frame with the mask
		# to form an output visualization
		#clone = np.squeeze(clone.cpu().numpy(), axis=0)

		output = ((0.3 * clone) + (0.7 * mask)).astype("uint8")
		
		#output = np.concatenate((clone, mask), axis=1).astype("uint8")
		# check if the video writer is None
		if writer is None:
	 		# initialize our video writer
	 		fourcc = 0x00000021
	 		writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,
	 			(output.shape[1], output.shape[0]), True)

	 		# some information on processing single frame
	 		if total > 0:
	 			elap = (end - start)
	 			with lock:
					 fps = elap
					 total_time = elap * total

	# 	# write the output frame to disk
		writer.write(output)

	# # check to see if we should display the output frame to our screen
		if SHOW > 0:
	 		cv2.imshow("Frame", output)
	 		key = cv2.waitKey(1) & 0xFF
 
	 	# if the `q` key was pressed, break from the loop
	 		if key == ord("q"):
	 			break
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()

	
	
			
@app.route("/output")
def get_result():
	return jsonify({
		'fps' : fps,
		"total_time" : total_time, 
	})
# release the file pointers
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True, help='ip address of the device')
	ap.add_argument("-o", "--port", type=int, required=True, help='ephemeral port number of the server (1024 to 65535)')

	
	
	args = vars(ap.parse_args())
	t = threading.Thread(target=segment)
	#segment()
	t.daemon = True
	t.start()
	app.run(host=args['ip'],port=args['port'], debug=True, threaded=True, use_reloader=False)



