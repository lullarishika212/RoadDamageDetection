import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image
from io import BytesIO
from datetime import datetime
import torch
import seaborn as sn
import numpy as np
# import pathlib
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt
import warnings
import time
import requests
import json
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from bokeh.io import show
from streamlit_bokeh_events import streamlit_bokeh_events
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from streamlit_js_eval import get_geolocation

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'models/trained_model/content/training_demo/exported_models'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'models/trained_model/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.10)

st.title("Road Damage Detection using Deep Learning Techniques")
st.header('YOLOv5\t	  Faster RCNN\t    	SSD')

STYLE = """
<style>
img {
    max-width: 100%;
}

div.stButton > button:first-child {
	background-color: #FF4B4B;
	color:white;
	font-size:20px;
	height:3em;
	width:8em;
	align: center;
	border-radius:10px 10px 10px 10px;
}
</style>
"""

# st.info(__doc__)
# st.markdown(STYLE, unsafe_allow_html=True)
# st.info(__doc__)

st.markdown(STYLE, unsafe_allow_html=True)

def file_uploader():
	file = st.file_uploader("Choose an image", type=["jpeg", "png", "jpg"], accept_multiple_files=False)
	
	# if not file:
	# 	st.text("Please upload a file of type: " + ", ".join(["jpeg", "png", "jpg"]))

	# if file is not None:
	return file

	# call Model prediction--

@st.cache(allow_output_mutation=True)
def load_model_yolo():
	model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
	return model1

@st.cache(allow_output_mutation=True)
def load_models():
	path_to_frcnn = PATH_TO_MODEL_DIR + '/my_model1/saved_model'
	path_to_ssd = PATH_TO_MODEL_DIR + '/my_model/saved_model'
	model2 = tf.saved_model.load(path_to_frcnn)
	model3 = tf.saved_model.load(path_to_ssd)
	models = [model2, model3]
	return models

def predict_yolo(file, imgpath):
	model1 = load_model_yolo()
	outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
	with open(imgpath, mode="wb") as f:
		f.write(file.getbuffer())

	model1.cpu()
	pred = model1(imgpath)
	pred.render()  # render bbox in image
	
	for im in pred.ims:
		im_base64 = Image.fromarray(im)
		im_base64.save(outputpath)

		img_ = Image.open(outputpath)
		st.image(img_, caption='YOLOv5', use_column_width='auto')


# 	# --Display predicton

	

	# IMAGE_PATHS = 'data/uploads/' + str(ts) + file.name

	

	# print('Loading model...', end='')
	# start_time = time.time()

	# # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
	# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

	# end_time = time.time()
	# elapsed_time = end_time - start_time
	# print('Done! Took {} seconds'.format(elapsed_time))

	# LOAD LABEL MAP DATA FOR PLOTTING
def predict_fr_ssd(imgpath, model, score):
	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
																		use_display_name=True)
	# models = load_models()

	def load_image_into_numpy_array(path):
		return np.array(Image.open(path))

	# print('Running inference for {}... '.format(IMAGE_PATHS), end='')

	image = cv2.imread(imgpath, 1)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_expanded = np.expand_dims(image_rgb, axis=0)

	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis, ...]

	# input_tensor = np.expand_dims(image_np, 0)
	detections = model(input_tensor)
	# detections2 = models[0](input_tensor)

	# All outputs are batches tensors.
	# Convert to numpy arrays, and take index [0] to remove the batch dimension.
	# We're only interested in the first num_detections.
	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy()
				for key, value in detections.items()}
	detections['num_detections'] = num_detections
	#print(detections)

	# detection_classes should be ints.
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	image_with_detections = image.copy()

	# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
	viz_utils.visualize_boxes_and_labels_on_image_array(
		image_with_detections,
		detections['detection_boxes'],
		detections['detection_classes'],
		detections['detection_scores'],
		category_index,
		use_normalized_coordinates=True,
		max_boxes_to_draw=200,
		min_score_thresh=score,
		agnostic_mode=False)

	return image_with_detections


def exe_detection(img_file, imgpath):
	# yolo = load_model_yolo()
	
	predict_yolo(img_file, imgpath)
	models = load_models()
	
	detect1 = predict_fr_ssd(imgpath, models[0], 0.4)
	detect2 = predict_fr_ssd(imgpath, models[1], 0.25)
	st.image(detect1, caption='FRCNN', use_column_width='auto')
	st.image(detect2, caption='SSD', use_column_width='auto')


img_file = file_uploader()

if img_file is not None:
	img = Image.open(img_file)
	#col1, col2, col3 = st.columns(3)
	st.image(img, caption="Uploaded image", use_column_width='auto')
	ts = datetime.timestamp(datetime.now())
	imgpath = os.path.join('data/uploads', str(ts) + img_file.name)
	submit = st.button("Detect")
	if submit:
		key = "4830d5bc966c4671a877cfb153d9cf4e"
		response = requests.get("https://api.ipgeolocation.io/ipgeo?apiKey=" + key)
		result = response.json()
		print(result['latitude'], result['city'], result['district'])
		# loc_button = Button(label="Get Location")
		# loc_button.js_on_event("button_click", CustomJS(code="""
		# 	navigator.geolocation.getCurrentPosition(
		# 		(loc) => {
		# 			document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
		# 		}
		# 	)
		# 	"""))
		# result = streamlit_bokeh_events(
		# 	loc_button,
		# 	events="GET_LOCATION",
		# 	key="get_location",
		# 	refresh_on_update=False,
		# 	override_height=75,
		# 	debounce_time=0)
		# show(loc_button)

		# if result:
		# 	if "GET_LOCATION" in result:
		# 		print(result.get("GET_LOCATION"))
		# loc = get_geolocation()
		# print(loc)
		# send_url = "http://api.ipstack.com/check?access_key=aac63cb1b8191dea1a1df865845d7670"
		# geo_req = requests.get(send_url)
		# geo_json = json.loads(geo_req.text)
		# print(geo_json)
		# latitude = geo_json['latitude']
		# longitude = geo_json['longitude']
		# city = geo_json['city']
		## region = geo_json['region_name']
		# st.text("Your coordinates are",latitude,",",longitude) 
		# st.text("Region:",region,"\tCity:", city)
		exe_detection(img_file, imgpath)

else:
	st.text("Please upload a file of type: " + ", ".join(["jpeg", "png", "jpg"]))