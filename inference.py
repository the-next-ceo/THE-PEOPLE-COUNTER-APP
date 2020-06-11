import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        #Initialize any class variables desired ###
        self.network = None
        self.plugin = None
        self.exec_network = None
        self.input_blob = None
        self.out_blob = None
        self.infer_request = None


    def load_model(self,model,device,cpu_extension):
        # Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize the plugin
        self.plugin = IECore()
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        # Add any necessary extensions ###
        
        # Check for supported layers ###
        # Check for any unsupported layers, and let the user 
        # know if anything is missing. Exit the program, if so
        supported_layers = self.plugin.query_network(self.network,device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether  cpu  extensions are available to add to IECore.")
            sys.exit(1)
            #self.plugin.add_extension(cpu_extension, device)
           
        # Return the loaded inference plugin ###
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network , device)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return self.exec_network;

    def get_input_shape(self):
        # Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self,frame):
        #tart an asynchronous request ###
        self.exec_network.start_async(request_id=0,inputs={self.input_blob: frame})
        return self.exec_network

    def wait(self):
        # Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        #Extract and return the output results
        return self.exec_network.requests[0].outputs[self.output_blob]
