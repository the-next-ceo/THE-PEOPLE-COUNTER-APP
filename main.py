"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output", type=str,
                        help="path to optional output video file")
    return parser
def draw_detection_boxes(frame, result, prob_th, width, height):
    #Draw bounding boxes onto the frame.
    for box in result[0][0]:
        confidence = box[2]
        if confidence > prob_th:
            xmin,xmax = map(lambda b : int(b*width), [box[3],box[5]])
            ymin,ymax = map(lambda b : int(b*height), [box[4],box[6]])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0) , 2)
            font = cv2.FONT_ITALIC
            info = 'PERSON DETECTED'
            display = cv2.putText(frame, info, (0, 130), font, 1, (0, 0 , 255))
            cv2.imshow('', display)
            
    return frame

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    PathArgs = args.input
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    get_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    if PathArgs == True:
        print("[INFO] opening video file...")
    single_image = False
    if PathArgs == 'CAM':
        PathArgs = 0
    elif PathArgs.endswith('.jpg')or PathArgs.endswith('.bmp'):
        single_image = True
    capture_on = cv2.VideoCapture(PathArgs)
    capture_on.open(PathArgs)
    #giving some input for width as "w" and height as "h"
    w = int(capture_on.get(3))
    h = int(capture_on.get(4))
    #intialising some variables
    current_count = 0  
    total_count = 0
    last_count = 0
    start_time = 0 
    delete = 0
    list_of = 0
    request_id=0
    
    
    
    
    ### TODO: Loop until stream is over ###
    while capture_on.isOpened():
   
        
        #if args.input is not None and writer is None:
            #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #writer = cv2.VideoWriter(args.input,fourcc,30,(w, h),True)
        if PathArgs is None:
            log.error("Warning ! Video  path not supplied\n")
            

        ### TODO: Read from the video capture ###
        flag, frame = capture_on.read()
        if not flag:
            break
        key = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        img_frame = cv2.resize(frame, (get_shape[3], get_shape[2]))
        img_frame = img_frame.transpose((2,0,1))
        img_frame = img_frame.reshape(1, *img_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(img_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait == 0:
            get_time = time.time() - inf_start
            

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results
            output = draw_detection_boxes(frame, result, prob_threshold,w,h)
            """"counter = 0
            
            extract = result[0, 0, :, 2]
            for p2, p1 in enumerate(extract):
                if p1 > prob_threshold:
                    counter = counter + 1
                    rectangle = extract[0, 0, p2, 3:]
                    q = (int(rectangle[0] * w), int(rectangle[1] * h))
                    r = (int(rectangle[2] * w), int(rectangle[3] * h))
                    frame = cv2.rectangle(frame, q, r, (0, 102, 0), 3)
                    font = cv2.FONT_ITALIC
                    info = 'PERSON DETECTED'
                    display = cv2.putText(frame, info, (0, 130), font, 1, (0, 0 , 255))
                    cv2.imshow('', display)"""
            font = cv2.FONT_ITALIC
            message = "2nd attempt | Inference time: {:.3f}ms"\
                               .format(get_time * 1000)
            cv2.putText(frame, message, (15,15), font,0.55,(0, 102, 0), 1)
            ### TODO: Calculate and send relevant information on ###
             ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > last_count:
                start_time = time.time()
                total_count += current_count - last_count
                
                
            # Person duration in the video calculation
            if current_count < last_count:
                #here converting the float duration to integer by using in below statement "int"
                duration = int(time.time() - start_time)
                if duration >=2:
                    total_count = total_count
                else:
                    total_count = total_count - last_count
                    
                
                client.publish(topic = "person", payload = json.dumps({"total": total_count}))
                list_of = total_count
                if duration >=2:
                    client.publish(topic ="person/duration", payload = json.dumps({"duration": duration}))
                    print('duration_terminal',duration)
            
            client.publish(topic = "person",payload = json.dumps({"count": current_count}))
            last_count = current_count
            if key == 27:
                break
           
        
        ### TODO: Send the frame to the FFMPEG server ###
        #frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
            cv2.imwrite('output_image.jpg', frame)
        
     
    capture_on.release()
    cv2.destroyAllWindows()
    #disconnecting from MQTT
    client.disconnect()
       
        


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
