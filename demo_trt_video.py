import sys
import os
import time
import argparse
import numpy as np
import cv2
# from PIL import Image
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import struct # !!!!!!!!!!!!!!!!!!
#from tool.utils import plot_boxes_cv2, load_class_names
import ctypes


def allocate_buffers2(engine, batch_size):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]

    #binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)  
        print("max batch_size", engine.max_batch_size, size) # 1 038 336
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0: size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print("size", size)
        # h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        # h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        # # Allocate device memory for inputs and outputs.
        # d_input = cuda.mem_alloc(h_input.nbytes)
        # d_output = cuda.mem_alloc(h_output.nbytes)
        # # Create a stream in which to copy inputs/outputs and run inference.
        # stream = cuda.Stream()
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
    #     # Allocate host and device buffers
    #     host_mem = cuda.pagelocked_empty(size, dtype)
    #     device_mem = cuda.mem_alloc(host_mem.nbytes)
    #     # Append the device buffer to device bindings.
    #     bindings.append(int(device_mem))
    #     # Append to the appropriate list.
    #     if engine.binding_is_input(binding): inputs.append(HostDeviceMem(host_mem, device_mem))
    #     else: outputs.append(HostDeviceMem(host_mem, device_mem))
    # return inputs, outputs, bindings, stream

def post_processing(cls, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_max_id = max_id[i, argwhere]
        #print("l_max_id",l_max_id)
        bboxes = []
        # nms for each class
        j = cls
        if cls in l_max_id:
        #for j in range(num_classes):
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)

    #t3 = time.time()

    #print('-----------------------------------')
    #print('       max and argmax : %f' % (t2 - t1))
    #print('                  nms : %f' % (t3 - t2))
    #print('Post processing total : %f' % (t3 - t1))
    #print('-----------------------------------')
    
    return bboxes_batch


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)


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
    #cuda.memcpy_htod_async(d_input, h_input, stream)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    #cuda.memcpy_dtoh_async(h_output, d_output, stream)
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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class GpuDevice(object):
    def __init__(self):
      #threading.Thread.__init__(self)
      self.skip_counter = 0
      #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
      #trt.init_libnvinfer_plugins(TRT_LOGGER, '')
      #self.trt_runtime = trt.Runtime(TRT_LOGGER)
      self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
      self.counter = 0
      self.cap = cv2.VideoCapture('39.avi')
      self.img_size = 416
      self.max_batch_size = 1
      self.image_size = (self.img_size, self.img_size)
      self.engine_path = 'yolov4_-1_3_'+str(self.img_size)+'_'+str(self.img_size)+'_dynamic.engine' # TRT inference time: 0.021343
      #self.trt_runtime = trt.Runtime(self.TRT_LOGGER)
      self.engine = None
      with open(self.engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
        self.engine = runtime.deserialize_cuda_engine(f.read())
            # Allocate buffers
        #print('Allocating Buffers')
      self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, self.max_batch_size)
      #print("1--",self.bindings,self.stream.handle)
      with self.engine.create_execution_context() as context:
        context.set_binding_shape(0, (self.max_batch_size, 3, self.img_size, self.img_size))
        while True:
          r, frame = self.cap.read()
          if (not r):
              print("skip frame ", self.skip_counter)
              self.skip_counter -= 1
              if (self.skip_counter > 0): continue
              else: break
          start = time.time()
          self.counter += 1
          frame = cv2.resize(frame,(self.img_size, self.img_size))
          frame_tf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
          #frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.uint8)/255.0
          frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.float32)/255.0
          frames = np.stack([frame_tf], axis=0)
          cur_batch_size = len(frames)
          frames = np.ascontiguousarray(frames)
          #print("Shape of the network input: ", frames.shape)
          #trt_outputs = detect2(context, inputs, outputs, bindings, stream, frames, cur_batch_size)
          self.inputs[0].host = frames
          #print("inputs",len(self.inputs[0].host))
          #print("outputs",len(self.outputs))
          trt_outputs = do_inference_new(context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=cur_batch_size)
          trt_outputs[0] = trt_outputs[0].reshape(cur_batch_size, -1, 1, 4)
          trt_outputs[1] = trt_outputs[1].reshape(cur_batch_size, -1, 80)
          # Output inference time
          #print("TensorRT inference time: {} ms".format(int(round((time.time() - inference_start_time) * 1000))))
          boxes = post_processing(0, 0.4, 0.6, trt_outputs)
          print(self.counter, time.time()-start, "boxes",  len(boxes[0]))
          if self.counter > 10:
            break
      self.cap.release()
      print("Finished OK")
      # self.kill()

def do_inference_new(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
    
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def main():
  cap = cv2.VideoCapture('39.avi')
  skip_counter = 0
  #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  counter = 0
  img_size = 416
  max_batch_size = 2
  image_size = (img_size, img_size)
  engine_path = 'yolov4_-1_3_'+str(img_size)+'_'+str(img_size)+'_dynamic.engine' # TRT inference time: 0.021343
  #self.engine = get_engine(self.engine_path, self.TRT_LOGGER)
  trt_runtime = trt.Runtime(TRT_LOGGER)
  engine = 0
  with open(engine_path, "rb") as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())
  context = engine.create_execution_context()
  context.set_binding_shape(0, (max_batch_size, 3, img_size, img_size))
  print("context", context)
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()
  for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        print("size", size)
        dims = engine.get_binding_shape(binding)        
        if dims[0] < 0: size *= -1        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print("trt dtype", dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        print("device_mem",device_mem)
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding): 
          inputs.append(HostDeviceMem(host_mem, device_mem))
        else: 
          outputs.append(HostDeviceMem(host_mem, device_mem))
  print("engine.get_binding_shape(binding)",engine.get_binding_shape(binding))
  # print("self.inputs")
  # print(inputs)
  # print("self.outputs")
  # print(outputs)
  # print("------------")

  #buffers = allocate_buffers(engine, max_batch_size)
  
  while True:
      r, frame = cap.read()
      if (not r):
          print("skip frame ", skip_counter)
          skip_counter -= 1
          if (skip_counter > 0): continue
          else: break
      start = time.time()
      counter += 1
      frame = cv2.resize(frame,(img_size, img_size))
      frame_tf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
      #frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.uint8)/255.0
      frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.float32)/255.0
      frames = np.stack([frame_tf], axis=0)
      cur_batch_size = len(frames)
      frames = np.ascontiguousarray(frames)
      #print("Shape of the network input: ", frames.shape)
      #trt_outputs = detect2(context, inputs, outputs, bindings, stream, frames, cur_batch_size)
      inputs[0].host = frames
      #np.copyto(inputs[0].host, frames)
      print("inputs",len(inputs[0].host))
      print("outputs",len(outputs))
      trt_outputs = do_inference_new(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=cur_batch_size)

      #print("to cpu ok", len(trt_outputs[0]))
      #print('Len of outputs: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
      trt_outputs[0] = trt_outputs[0].reshape(cur_batch_size, -1, 1, 4)
      trt_outputs[1] = trt_outputs[1].reshape(cur_batch_size, -1, 80)

      #print("trt_outputs: ", type(trt_outputs), len(trt_outputs), type(trt_outputs[0]), len(trt_outputs[0]), trt_outputs[0].shape, trt_outputs[1].shape)
      boxes = post_processing(0, 0.4, 0.6, trt_outputs)
      print("boxes", counter, len(boxes[0]))
      if counter > 1:
        break
  print("end")

  if context:
      context.pop()
  del context
  if cap:
    cap.release()

def main1():
  cap = cv2.VideoCapture('39.avi')
  skip_counter = 0
  #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  counter = 0
  img_size = 416
  max_batch_size = 2
  image_size = (img_size, img_size)
  engine_path = 'yolov4_-1_3_'+str(img_size)+'_'+str(img_size)+'_dynamic.engine' # TRT inference time: 0.021343
  #self.engine = get_engine(self.engine_path, self.TRT_LOGGER)
  trt_runtime = trt.Runtime(TRT_LOGGER)
  engine = 0
  #engine = get_engine(engine_path)
  with open(engine_path, "rb") as f:
     engine = trt_runtime.deserialize_cuda_engine(f.read())
  context = engine.create_execution_context()
  context.set_binding_shape(0, (max_batch_size, 3, img_size, img_size))

  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Device(0).Stream() #!!!!!!!!!!!
  for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        print("size", size)
        dims = engine.get_binding_shape(binding)        
        if dims[0] < 0: size *= -1        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print("trt dtype", dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        print("device_mem",device_mem)
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding): 
          inputs.append(HostDeviceMem(host_mem, device_mem))
        else: 
          outputs.append(HostDeviceMem(host_mem, device_mem))
  #buffers = inputs, outputs, bindings, stream
  #buffers = allocate_buffers(engine, max_batch_size)

  while True:
      r, frame = cap.read()
      if (not r):
          print("skip frame ", skip_counter)
          skip_counter -= 1
          if (skip_counter > 0): continue
          else: break
      start = time.time()
      counter += 1
      frame = cv2.resize(frame,(img_size, img_size))
      frame_tf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
      #frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.uint8)/255.0
      frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.float32)/255.0
      frames = np.stack([frame_tf,frame_tf], axis=0)
      cur_batch_size = len(frames)
      frames = np.ascontiguousarray(frames)
      print("Shape of the network input: ", frames.shape)
      #trt_outputs = detect2(context, inputs, outputs, bindings, stream, frames, cur_batch_size)
      inputs[0].host = frames
      print("inputs[0].host",len(inputs[0].host))
      trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
      #print('Len of outputs: ', len(trt_outputs), trt_outputs[0].shape, trt_outputs[1].shape)
      trt_outputs[0] = trt_outputs[0].reshape(cur_batch_size, -1, 1, 4)
      trt_outputs[1] = trt_outputs[1].reshape(cur_batch_size, -1, 80)
      print("trt_outputs: ", len(trt_outputs[0]), trt_outputs[0].shape)
      #print("dd", trt_outputs[0][0].shape, trt_outputs[1][0].shape)
      #tr_0 = [[trt_outputs[0][0], trt_outputs[1][0]]]
      #print("tr", tr_0[0].shape, tr_0[1].shape)
      boxes = post_processing(0, 0.4, 0.6, trt_outputs)
      print("people", len(boxes[0]))
      #boxes_1 = post_processing(images[1], 0.4, 0.6, [trt_outputs[1], trt_outputs[1])
      #plot_boxes_cv2(image_src_0, boxes[0], savename='01_trt2.jpg', class_names=class_names)
      #plot_boxes_cv2(image_src_1, boxes[1], savename='02_trt2.jpg', class_names=class_names)
      print("counter", counter)
      if counter > 3:
        break      

def main2():
    image_size = (416, 416)
    engine_path = 'yolov4_-1_3_416_416_dynamic.engine' # TRT inference time: 0.021343
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
    images = np.ascontiguousarray(images)    
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
        boxes = post_processing(22, 0.4, 0.6, trt_outputs)
        print("boxes", type(boxes[0]), boxes[0])
        #boxes_1 = post_processing(images[1], 0.4, 0.6, [trt_outputs[1], trt_outputs[1])
        #plot_boxes_cv2(image_src_0, boxes[0], savename='01_trt2.jpg', class_names=class_names)
        #plot_boxes_cv2(image_src_1, boxes[1], savename='02_trt2.jpg', class_names=class_names)


def main3():
    image_size = (416, 416)
    engine_path = 'yolov4_-1_3_416_416_dynamic.engine' # TRT inference time: 0.021343
    engine = get_engine(engine_path)
    context = engine.create_execution_context()
    size = 2
    buffers = allocate_buffers(engine, size)
    context.set_binding_shape(0, (size, 3, image_size[0], image_size[1]))

    for i in range(3):
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
      images = np.ascontiguousarray(images)    

      print("Shape of the network input: ", images.shape)
      trt_outputs = detect(context, buffers, images, size)
      #print("trt_outputs: ", type(trt_outputs), len(trt_outputs), type(trt_outputs[0]), len(trt_outputs[0]), trt_outputs[0].shape, trt_outputs[1].shape)
      #print("dd", trt_outputs[0][0].shape, trt_outputs[1][0].shape)
      #tr_0 = [[trt_outputs[0][0], trt_outputs[1][0]]]
      #print("tr", tr_0[0].shape, tr_0[1].shape)
      boxes = post_processing(22, 0.4, 0.6, trt_outputs)
      print("boxes", type(boxes[0]), boxes[0])
      #boxes_1 = post_processing(images[1], 0.4, 0.6, [trt_outputs[1], trt_outputs[1])
      #plot_boxes_cv2(image_src_0, boxes[0], savename='01_trt2.jpg', class_names=class_names)
      #plot_boxes_cv2(image_src_1, boxes[1], savename='02_trt2.jpg', class_names=class_names)

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

def prep_img(image_src):    
    img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)/255.0
    return img_in

def detect2(context, inputs, outputs, bindings, stream, images, size=1):
    ta = time.time()
    #inputs, outputs, bindings, stream = buffers
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

#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#main()
gpu = 0
gpu = GpuDevice()
time.sleep(10)
print("stop by time")
# if gpu:
  # gpu.kill()
