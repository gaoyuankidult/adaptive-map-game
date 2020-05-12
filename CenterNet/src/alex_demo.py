from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
import json

from opts import opts
from detectors.detector_factory import detector_factory
from tobiiglassesctrl import TobiiGlassesController 

import concurrent.futures
import datetime

# import custom definitions
from utils import *

log_data = True

info = {'focus': {
        'screen': {'gaze_focused':False, 'probability':None},
        'tablet': {'gaze_focused':False, 'probability':None},
        'robot': {'gaze_focused':False, 'probability':None}}}

screen_criteria = 950000
colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]

def add_coco_bbox(bbox, cat, label, img, conf=1, show_txt=True): 
  bbox = np.array(bbox, dtype=np.int32)
  cat = int(cat)
  c = colors[cat][0][0].tolist()
  txt = '{}{:.1f}'.format(label, conf)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
  cv2.rectangle(
    img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
  if show_txt:
    cv2.rectangle(img,
                  (bbox[0], bbox[1] - cat_size[1] - 2),
                  (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
    cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def demo(opt):
  def find_ellipses_parallel(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, gray_frame = cv2.threshold(gray_frame, 170, 235, 0)  
    contours, hier = cv2.findContours(gray_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    elps = []
    for cnt in contours:
      try:
        elp = cv2.fitEllipse(cnt)
        elps.append(elp)
      except:
        pass
    return elps


  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  detector.pause = False
  wait_signal = False

  # Wait for the UDP message to start
  wait_start_signal = wait_signal
  if wait_start_signal:
    import socket
    from socket import gethostbyname
    UDP_PORT_NO = 9013
    UDP_IP_ADDRESS = gethostbyname("0.0.0.0")
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
    while True:
      print("Waiting for the start message...")
      data, addr = serverSock.recvfrom(1024)
      
      print ("Message: " + data.decode())
      if data.decode().startswith("START:"):
        _, game_id, session = data.decode().split(':')
        game_id = int(game_id)
        #session = int (session)
        print("Received game id %d."%(game_id))
        break

  # Decide the camera source 
  if opt.demo == 'tobii':
    ipv4_address = "192.168.0.109"
    cap = cv2.VideoCapture("rtsp://%s:8554/live/scene" %ipv4_address)
    tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)
    video_freq = tobiiglasses.get_video_freq()
    frame_duration = 1000.0/float(video_freq)
    tobiiglasses.start_streaming()
  elif opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cap = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  # Set up the threading mechanism for factobiiglasses.start_streaming() recognition and stopp signal receiving.
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: 
    # Wait for the UDP message to end
    wait_stopp_signal = wait_signal
    if wait_start_signal and wait_stopp_signal:
      def wait_stopp():
        while True:
          print("Waiting for the end message...")
          data, addr = serverSock.recvfrom(1024)
          
          print ("Message: " + data.decode())
          if data.decode().startswith("STOPP:"):
            break
        return
      udp_end_received = pool.submit(wait_stopp)
      
    ret, img = cap.read()
    future = pool.submit(find_ellipses_parallel,img)

    # Setup the HTTP patch method
    send_back = wait_signal
    if wait_start_signal and send_back:
      import requests
      import time
      ip = '192.168.0.58'
      port = ':3000'
      address = 'http://' + ip + port + '/game/'+str(game_id)+'/'

    save_log = True
    if wait_start_signal and save_log:
      infolog = open("log_"+str(game_id)+".txt","w+")
      infolog.write(str(datetime.datetime.now()))
      infolog.write("\n")
      infolog.write("START: ")
      infolog.write("\n")

    # Set up the face recognition
    detection_cnt = 0
    recognized = False
    MA = 0
    ma = 0
    elps = [] 
    head_degrees = np.array([0,0,0])

    screen_probability = -1.0
    tablet_probability = -1.0  
    
    while(cap.isOpened()):
      data_update_ready = False
      
      info["focus"]["robot"]["gaze_focused"] = False
      info["focus"]["screen"]["gaze_focused"] = False
      info["focus"]["tablet"]["gaze_focused"] = False

      info["focus"]["robot"]["probability"] = -1.0
      info["focus"]["screen"]["probability"] = -1.0
      info["focus"]["tablet"]["probability"] = -1.0

      #if opt.demo == 'tobii':
      #  data_gy = tobiiglasses.get_data()['mems']['gy']['gy']
        #print(data_gy)
        #nput()
      #  head_degrees = head_degrees + np.array(data_gy)
      #  print(head_degrees)


      # Detect all the things in the scene
      detection_cnt = detection_cnt + 1
      ret, img = cap.read()
      if detection_cnt %8== 0:
        height, width = img.shape[:2]
        if ret == True:

          # Detect everything
          centernet_results = detector.run(img)

          detect_robot= True
          if detect_robot:
            if future.done():
              elps = future.result() 
              future = pool.submit(find_ellipses_parallel,img)

          # Draw everyting
          draw_robot = True
          if draw_robot:
            margin = 2.6
            for elp in elps:
              (x,y), (MA,ma), angle = elp

              if 100 < MA < 440 and 100 <ma < 330 and 80264 <np.pi * ma *MA < 270000:
                cv2.ellipse(img, ((x,y), (MA*margin,ma*margin), angle), (0,255,0),3)

          detect_gaze = True
          if opt.demo == 'tobii' and detect_gaze:
            data_gp  = tobiiglasses.get_data()['gp']
            data_pts = tobiiglasses.get_data()['pts']
            data_gy  = tobiiglasses.get_data()['mems']['gy']
            data_ac = tobiiglasses.get_data()['mems']['ac']
            data_lpd = tobiiglasses.get_data()['left_eye']['pd']            
            data_rpd = tobiiglasses.get_data()['right_eye']['pd']

            offset = data_gp['ts']/1000000.0 - data_pts['ts']/1000000.0
            gaze_detected = offset > 0.0 and offset <= frame_duration
            if gaze_detected:
              cv2.circle(img,(int(data_gp['gp'][0]*width),int(data_gp['gp'][1]*height)), 30, (0,0,255), 2)
              gaze_x = int(data_gp['gp'][0]*width)
              gaze_y = int(data_gp['gp'][1]*height)

          results = centernet_results["results"]
          for j in [63,64]:
            for bbox in results[j]:
              if bbox[4] > detector.opt.vis_thresh:
                confidence = bbox[4]
                name = coco_class_name[j-1]
                if name == "laptop" or name == "tv":
                  if (bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2 < screen_criteria:
                    gaze_on_tablet = True
                    add_coco_bbox(bbox, j, "tablet", img, conf=confidence, show_txt=True)
                  else:
                    gaze_on_screen = True
                    add_coco_bbox(bbox, j, "screen", img, conf=confidence, show_txt=True)

      # Forming the sending informaiton
      send_numbers = True
      detect_gaze_on_robot = True
      if detection_cnt %8== 0:
        gaze_on_robot = False
        if opt.demo == 'tobii' and detect_gaze and gaze_detected and detect_robot and draw_robot and detect_gaze_on_robot and elps is not []:
          for elp in elps:
            (x,y), (MA,ma), angle = elp
            try:
              #if 80264 <np.pi * ma *MA < 270000:
              if 100 < MA < 440 and 100 <ma < 330 and 80264 <np.pi * ma *MA < 270000:
                gaze_on_robot = ((gaze_y-y)**2/(ma*margin)**2 +  (gaze_x-x)**2/(MA*margin)**2 <= 1)
            except ZeroDivisionError:
              print("Zero Division happened, probabolly no face detected or detected wrong area.")
        if gaze_on_robot:
            print("Gaze on robot")

        
        gaze_on_screen = False
        gaze_on_tablet = False
        detect_gaze_on_object = True

        if opt.demo == 'tobii' and detect_gaze and gaze_detected and not gaze_on_robot:
          results = centernet_results["results"]
          # tv63 and laptop 64
          for j in [63,64]:
            for bbox in results[j]:
              if bbox[4] > detector.opt.vis_thresh:
                confidence = bbox[4]
                name = coco_class_name[j-1]
                if name == "laptop" or name == "tv":
                  x_in_range = bbox[0] <= gaze_x <= bbox[2] if bbox[2] >= bbox[0] else bbox[2] <= gaze_x <= bbox[0]
                  y_in_range = bbox[1] <= gaze_y <= bbox[3] if bbox[3] >= bbox[1] else bbox[3] <= gaze_x <= bbox[1]
                  gaze_on_object = x_in_range and y_in_range
                  if gaze_on_object:
                    if (bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2 < screen_criteria:
                      gaze_on_tablet = True
                      tablet_probability = confidence
                    else:
                      gaze_on_screen = True
                      screen_probability = confidence
                      
          if gaze_on_tablet:
            print("Gaze on tablet")
          if gaze_on_screen:
            print("Gaze on screen")

        cv2.imshow("demo",img)
        
        if not gaze_on_robot and not gaze_on_screen and not gaze_on_tablet:
          print("Gaze on no where")


        if screen_probability != None and tablet_probability != None:
            data_update_ready = True
        print("data upgrade", data_update_ready)
        if send_numbers and data_update_ready: #and screen_probability == None and and tablet_coordinates  None and tablet_probability is not None:
          # Check whether the gaze is in the rectangle of the of the robot face
          # Start 

          info["focus"]["robot"]["gaze_focused"] = bool(gaze_on_robot)
          info["focus"]["screen"]["gaze_focused"] = bool(gaze_on_screen)
          info["focus"]["tablet"]["gaze_focused"] = bool(gaze_on_tablet)

          info["focus"]["robot"]["probability"] = -1.0
          info["focus"]["screen"]["probability"] = float(screen_probability)
          info["focus"]["tablet"]["probability"] = float(tablet_probability)
        
        if wait_start_signal and save_log:
          infolog.write(str(datetime.datetime.now()))
          infolog.write("\n")
          infolog.write(json.dumps(info))
          infolog.write("\n")
          infolog.write(json.dumps(data_gp))
          infolog.write("\n")
          infolog.write(json.dumps(data_gy))
          infolog.write("\n")
          infolog.write(json.dumps(data_ac))
          infolog.write("\n")
          infolog.write(json.dumps(data_lpd))
          infolog.write("\n")
          infolog.write(json.dumps(data_rpd))
          infolog.write("\n")                                

        if send_back:
          if not data_update_ready:
            print("Not updated")
          print(info)
          r = requests.patch(address,data=json.dumps(info))
        #print(r.status_code)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        return

      if wait_stopp_signal:
        if udp_end_received.done():
          if save_log:
            infolog.write(str(datetime.datetime.now()))
            infolog.write("\n")
            infolog.write("END: ")
            infolog.write("\n")
            infolog.close()
          cap.release()
          cv2.destroyAllWindows()
          return

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)


"""
for j in [63,64]:
  for bbox in results[j]:
    if bbox[4] > detector.opt.vis_thresh:
      confidence = bbox[4]
      name = coco_class_name[j-1]
      if name == "laptop" or name == "tv":
        add_coco_bbox(bbox, j, img, conf=1, show_txt=True)
"""

"""
  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
"""

"""              
              if detect_gaze_on_tablet and name == "laptop":
                x_in_range = bbox[0] <= gaze_x <= bbox[2] if bbox[2] >= bbox[0] else bbox[2] <= gaze_x <= bbox[0]
                y_in_range = bbox[1] <= gaze_y <= bbox[3] if bbox[3] >= bbox[1] else bbox[3] <= gaze_x <= bbox[1]
                tablet_coordinates = bbox[:4]
                tablet_probability = confidence
                gaze_on_table = x_in_range and y_in_range
                if gaze_on_table:
                  print("Gaze on tablet detected")
                else:
                  print("Gaze on  bablet not detected")
              if detect_gaze_on_screen and name == "tv":
                x_in_range = bbox[0] <= gaze_x <= bbox[2] if bbox[2] >= bbox[0] else bbox[2] <= gaze_x <= bbox[0]
                y_in_range = bbox[1] <= gaze_y <= bbox[3] if bbox[3] >= bbox[1] else bbox[3] <= gaze_x <= bbox[1]
                screen_coordinates = bbox[:4]
                screen_probability = confidence
                gaze_on_screen = x_in_range and y_in_range
                if gaze_on_screen:
                  print("Gaze on screen detected")
                else:
                  print("Gaze on screen not detected")
"""