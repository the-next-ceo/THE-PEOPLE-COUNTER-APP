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
    return parser


def connect_mqtt():
    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client
 
def draw_boxes(frame,output_result,prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    """
    counter=0
    # Start coordinate, here (xmin, ymin) 
    # represents the top left corner of rectangle 
    start_point = None
    # Ending coordinate, here (xmax, ymax) 
    # represents the bottom right corner of rectangle 
    end_point = None 
    # Deep blue color in BGR 
    color = (102, 51, 0)
    # Line thickness of 3 px 
    thickness = 3
    for box in output_result[0][0]: # Output shape is 1x1x100x7
        if box[2] > prob_threshold:
            start_point = (int(box[3] * width), int(box[4] * height))
            end_point = (int(box[5] * width), int(box[6] * height))
            # Using cv2.rectangle() method 
            # Draw a rectangle with Green line borders of thickness of 1 px
            frame = cv2.rectangle(frame, start_point, end_point, color,thickness)
            counter+=1
    return frame, counter
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Input arguments
    model = args.model
    device = args.device
    cpuExtension = args.cpu_extension
    probThresholdArgs = args.prob_threshold
    PathArgs = args.input
    
    # Initialise the class
    infer_network = Network()
    
    #Load the model through `infer_network`
    infer_network.load_model(model,device,cpuExtension)
    net_input_shape = infer_network.get_input_shape()
    
    # Set Probability threshold for detections
    prob_threshold = probThresholdArgs
    
    # Handle image, video or webcam
    # Create a flag for single images
    # Flag for the input image
    single_image_mode = False
    # Check if the input is a webcam
    if PathArgs == 'CAM':
        PathArgs = 0
    elif PathArgs.endswith('.jpg') or PathArgs.endswith('.bmp'):
        single_image_mode = True

    # Handle the input stream 
    # Get and open video capture
    capture = cv2.VideoCapture(PathArgs)
    capture.open(PathArgs)

    # Grab the shape of the input 
    width = int(capture.get(3))
    height = int(capture.get(4))
    
    # initlise some variable 
    info = 0
    counter = 0
    prev_counter = 0
    prev_time = 0
    total = 0
    dur = 0
    request_id=0
    
    # Process frames until the video ends, or process is exited
    while capture.isOpened():
        # Read the next frame
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        #Re-size the frame to inputshape_width x inputshape_height
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        #Start asynchronous inference for specified request
        #Perform inference on the frame
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(p_frame)
        # Get the output of inference
        if infer_network.wait() == 0:
            get_time = time.time() - inf_start
            # Results of the output layer of the network
            output_results = infer_network.get_output()
            #Extract any desired stats from the results 
            #Update the frame to include detected bounding boxes
            frame_made_of_rect, pointer = draw_boxes(frame, output_results, prob_threshold, width, height)
            #Display inference time
            inf_time_message = "THIS IS SHOWTIME | Inference time: {:.3f}ms"\
                               .format(get_time * 1000)
            cv2.putText(frame_made_of_rect, inf_time_message, (15, 15),
                       cv2.FONT_ITALIC, 0.55, (0, 102, 0), 1)
                    
            #Calculate and send relevant information on 
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if pointer != counter:
                prev_counter = counter
                counter = pointer
                if dur >= 3:
                    prev_time = dur
                    dur = 0
                else:
                    dur = prev_time + dur
                    prev_time = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    info = counter
                    if dur == 3 and counter > prev_counter:
                        total += counter - prev_counter
                    elif dur == 3 and counter < prev_counter:
                        duration_report = int((prev_time / 10.0) * 1000)
                        
            client.publish('person',
                           payload=json.dumps({
                               'count': info, 'total': total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
            
            #Send frame to the ffmpeg server
            #  Resize the frame
            #frame = cv2.resize(frame, (768, 432))
            sys.stdout.buffer.write(frame_made_of_rect)
            sys.stdout.flush()
            
            if single_image_mode:
                cv2.imwrite('output_image.jpg', frame_made_of_rect)

        # Break if escapturee key pressed
        if key_pressed == 27:
            break
        

    # Release the out writer, captureture, and destroy any OpenCV windows
    capture.release()
    cv2.destroyAllWindows()
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