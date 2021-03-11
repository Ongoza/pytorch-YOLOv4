import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *

def GiB(val):
    return val * 1 << 30

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0: size *= -1        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding): inputs.append(HostDeviceMem(host_mem, device_mem))
        else: outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

TRT_LOGGER = trt.Logger()

def prep_img(image_src):    
    img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)/255.0
    return img_in

def main():
    # print("GiB",GiB(1))
    image_size = (608, 608)

    namesfile = 'data/coco.names'
    class_names = load_class_names(namesfile)
    engine_path = 'yolov4_-1_3_608_608_dynamic.engine'
    
    image_path_0 = 'data/giraffe.jpg' 
    image_src_0 = cv2.imread(image_path_0)
    image_src_0 = cv2.resize(image_src_0, image_size, interpolation=cv2.INTER_LINEAR)
    img_in_0 = prep_img(image_src_0)

    image_path_1 = 'data/dog.jpg' 
    image_src_1 = cv2.imread(image_path_1)
    image_src_1 = cv2.resize(image_src_1, image_size, interpolation=cv2.INTER_LINEAR)
    img_in_1 = prep_img(image_src_1)

    print("img_in", img_in_0.shape)
    images = np.stack([img_in_0, img_in_1], axis=0)
    #images = [img_src, img_src]
    #img_in = np.expand_dims(img_in, axis=0)
    images = np.ascontiguousarray(images)
    #images = img_in
    size = 2
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, size)
        context.set_binding_shape(0, (size, 3, image_size[0], image_size[1]))
        print("Shape of the network input: ", images.shape)
        trt_outputs = detect(context, buffers, images, size)
        #print("trt_outputs: ", type(trt_outputs), len(trt_outputs), type(trt_outputs[0]), len(trt_outputs[0]), trt_outputs[0].shape, trt_outputs[1].shape)
        #print("dd", trt_outputs[0][0].shape, trt_outputs[1][0].shape)
        #tr_0 = [[trt_outputs[0][0], trt_outputs[1][0]]]
        #print("tr", tr_0[0].shape, tr_0[1].shape)
        boxes = post_processing(0, 0.4, 0.6, trt_outputs)
        #print("boxes", type(boxes[0]), boxes[0])
        #boxes_1 = post_processing(images[1], 0.4, 0.6, [trt_outputs[1], trt_outputs[1])
        plot_boxes_cv2(image_src_0, boxes[0], savename='01_trt2.jpg', class_names=class_names)
        plot_boxes_cv2(image_src_1, boxes[1], savename='02_trt2.jpg', class_names=class_names)

def detect(context, buffers, images, size=1):
    ta = time.time()
    inputs, outputs, bindings, stream = buffers
    #print('Length of inputs: ', len(inputs))
    inputs[0].host = images
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #print('Len of outputs: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
    trt_outputs[0] = trt_outputs[0].reshape(size, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(size, -1, 80)
    #print('Len of outputs 2: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
    tb = time.time()
    print('-----------------------------------')
    print('    TRT inference time: %f' % (tb - ta))
    print('-----------------------------------')
    return trt_outputs


if __name__ == '__main__':
  main()
