# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import ctypes
import threading #Write a multithreading parallelization module

#drone GPS data get
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
import FlightFunc
import time

from initmap import initmap
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [15, 8]
mpl.rcParams['savefig.dpi'] = 100
global vehicle
connection_string = "tcp:127.0.0.1:5762"
vehicle = connect(connection_string, baud=115200, wait_ready=True)
m = 0
j = 0
k = 0
o = 0
draw_dolosse_n = 0
def drone_Route():
    # Main
    global vehicle
    # Download the missionlist
    cmd = vehicle.commands
    cmd.download()
    cmd.wait_ready()
    time.sleep(1)
    print("Clean the commandlist ...")
    cmd.clear()
    
    
    # Takeoff command in GUIDED
    FlightFunc.arm_and_takeoff(vehicle,9)
    
    # hordaine 0
    msg = Command(
    	0, 0, 0,
    	mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    	mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    	0, 0, 0, 0, 0, 0, 25.151320, 121.782376, 10) # lat, lon, alt
    cmd.add(msg)
    # jonshon 1
    msg = Command(
    	0, 0, 0,
    	mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    	mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    	0, 0, 0, 0, 0, 0, 25.151440, 121.781974, 10) # lat, lon, alt
    cmd.add(msg)
    # point3 2
    msg = Command(
    	0, 0, 0,
    	mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    	mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    	0, 0, 0, 0, 0, 0, 25.151557, 121.781618, 10) # lat, lon, alt
    cmd.add(msg)
    # End point 3
    msg = Command(
    	0, 0, 0,
    	mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    	mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
    	0, 0, 0, 0, 0, 0, 25.151557, 121.781618, 10) # lat, lon, alt
    cmd.add(msg)
    
    # Upload the command
    print("Uploading the mission ... ")
    cmd.upload()
    
    # Set speed
    vehicle.groundspeed = 2.2
    
    # Reset the mission set to first waypoint ,and start the mission 
    vehicle.commands.next = 0
    time.sleep(93)
    vehicle.mode = VehicleMode("AUTO")
    
    while True:
    	next_waypoint = vehicle.commands.next    
    	print("There are %s waypoint and ..." %len(vehicle.commands))
    	print("NOW GOING TO WAYPOINT %s" %next_waypoint)    
    #	if condition == 1:
    #		print(" Detected !")
    #		vehicle.mode = VehicleMode("GUIDED")
    #		#FlightFunc.send_ned_velocity(vehicle,velocity[0],velocity[1],velocity[2],velocity[3])  
    #		time.sleep(5)
    #		detect = 0      
    #		vehicle.mode = VehicleMode("AUTO")
    #		time.sleep(10)
    		
    	if next_waypoint==4 :
    		print("Mission complete ! ")
    		break
    	time.sleep(1)
    
    vehicle.mode = VehicleMode("RTL")
    time.sleep(0.5)
    print("RTL")
    vehicle.close()

def buzzer():
    ctypes.windll.kernel32.Beep(1000,1000) #0.5 second beep



drone_Route_work = threading.Thread(target = drone_Route)
drone_Route_work.start()
#time.sleep(20)
center_lon = vehicle.location.global_frame.lon
center_lat = vehicle.location.global_frame.lat
flight_map = initmap(center_lon, center_lat)
flight_map.createmap(center_lon, center_lat) 

#center_point_change90 = {"[0, 0]":"[0, 4]", "[0, 1]":"[1, 4]", "[0, 2]":"[2, 4]", "[0, 3]":"[3, 4]", "[0, 4]":"[4, 4]", "[1, 0]":"[0, 3]", "[1, 1]":"[1, 3]", "[1, 2]":"[2, 3]", "[1, 3]":"[3, 3]", "[1, 4]":"[4, 3]", "[2, 0]":"[0, 2]", "[2, 1]":"[1, 2]", "[2, 2]":"[2, 2]", "[2, 3]":"[3, 2]", "[2, 4]":"[4, 2]", "[3, 0]":"[0, 1]", "[3, 1]":"[1, 1]", "[3, 2]":"[2, 1]", "[3, 3]":"[3, 1]", "[3, 4]":"[4, 1]", "[4, 0]":"[0, 0]", "[4, 1]":"[1, 0]", "[4, 2]":"[2, 0]", "[4, 3]":"[3, 0]", "[4, 4]":"[4, 0]"}

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.7,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        global m
        global j
        global k
        global o
        global draw_dolosse_n
        start = timer()
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        draw_dolosse_n+=1
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            GPS_lon = vehicle.location.global_frame.lon
            GPS_lat = vehicle.location.global_frame.lat
            center_point = [int(((left+right)/2-1)/172),int(((top+bottom)/2-1)/96)] #[The larger the object position and the more left,The larger the object position and the lower the position]
#            center_point_90 = center_point_change90[str(center_point)]
#            print("center_point:[0]"+str(center_point[0]))
#            print("center_point:[1]"+str(center_point[1]))
            if((label.split(' ')[0] == 'dolosse') & ((draw_dolosse_n%30)==0)):
#                draw_dolosse_work = threading.Thread(target = flight_map.draw_dolosse,args = (center_lon, center_lat,GPS_lon+(center_point[0]-2)*0.00025,GPS_lat+(center_point[1]-2)*0.00025))

                draw_dolosse_work = threading.Thread(target = flight_map.draw_dolosse,args = (center_lon, center_lat,GPS_lon+((int(center_point[0])+0) if (int(center_point[1])) >3 else int(center_point[0])-2 )*0.0002,GPS_lat+(int(center_point[1])-3)*0.00025))
                draw_dolosse_work.start()
                time.sleep(0.01)                      
#               flight_map.draw_dolosse(center_lon, center_lat,GPS_lon+(center_point[0]-2)*0.00025,GPS_lat+(center_point[1]-2)*0.00025)
            elif((label.split(' ')[0] == 'Human1') & (m==0)):
                buzzer_work = threading.Thread(target = buzzer)
                buzzer_work.start()
#                                ctypes.windll.kernel32.Beep(1000,300) #0.3 second beep
                draw_person_work = threading.Thread(target = flight_map.draw_person,args = (center_lon, center_lat,GPS_lon+(int(center_point[0])-2)*0.0002,GPS_lat+(int(center_point[1])-2)*0.0002))
                draw_person_work.start()
                m+=1
            elif((label.split(' ')[0] == 'Human2') & (j==0)):
                buzzer_work = threading.Thread(target = buzzer)
                buzzer_work.start()
#                                ctypes.windll.kernel32.Beep(1000,300) #0.3 second beep
                draw_person_work = threading.Thread(target = flight_map.draw_person,args = (center_lon, center_lat,GPS_lon+(int(center_point[0])-2)*0.0002,GPS_lat+(int(center_point[1])-2)*0.0002))
                draw_person_work.start()
                j+=1
            elif((label.split(' ')[0] == 'Human3') & (k==0)):
                buzzer_work = threading.Thread(target = buzzer)
                buzzer_work.start()
#                                ctypes.windll.kernel32.Beep(1000,300) #0.3 second beep
                draw_person_work = threading.Thread(target = flight_map.draw_person,args = (center_lon, center_lat,GPS_lon+(int(center_point[0])-2)*0.0002,GPS_lat+(int(center_point[1])-2)*0.0002))
                draw_person_work.start()
                k+=1
            elif((label.split(' ')[0] == 'Human4') & (o==0)):
                buzzer_work = threading.Thread(target = buzzer)
                buzzer_work.start()
#                                ctypes.windll.kernel32.Beep(1000,300) #0.3 second beep
                draw_person_work = threading.Thread(target = flight_map.draw_person,args = (center_lon, center_lat,GPS_lon+(int(center_point[0])-2)*0.0002,GPS_lat+(int(center_point[1])-4)*0.0002))
                draw_person_work.start()
                o+=1                    
                    
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print("物件偵測時長:"+str(end - start))
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def mat_inter(box1,box2):
    # 判断兩矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
 
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return 1 #True
    else:
        return 0 #False