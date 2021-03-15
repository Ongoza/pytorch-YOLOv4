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

from tool.utils import plot_boxes_cv2, load_class_names

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

    t3 = time.time()

    print('-----------------------------------')
    print('       max and argmax : %f' % (t2 - t1))
    print('                  nms : %f' % (t3 - t2))
    print('Post processing total : %f' % (t3 - t1))
    print('-----------------------------------')
    
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

def main():
  cap = cv2.VideoCapture('39.avi')
  skip_counter = 0
  counter = 0
  img_size = 416
  max_batch_size = 2
  image_size = (img_size, img_size)
  engine_path = 'yolov4_-1_3_'+str(img_size)+'_'+str(img_size)+'_dynamic.engine' # TRT inference time: 0.021343

  engine = get_engine(engine_path)
  context = engine.create_execution_context()
  buffers = allocate_buffers(engine, max_batch_size)
  context.set_binding_shape(0, (max_batch_size, 3, img_size, img_size))

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
    frame_tf = np.transpose(frame_tf, (2, 0, 1)).astype(np.float32)/255.0
    frames = np.stack([frame_tf], axis=0)
    cur_batch_size = len(frames)
    frames = np.ascontiguousarray(frames)
    print("Shape of the network input: ", frames.shape)
    trt_outputs = detect(context, buffers, frames, cur_batch_size)
    print("trt_outputs: ", type(trt_outputs), len(trt_outputs), type(trt_outputs[0]), len(trt_outputs[0]), trt_outputs[0].shape, trt_outputs[1].shape)
    boxes = post_processing(0, 0.4, 0.6, trt_outputs)
    print("boxes", type(boxes[0]), len(boxes[0]))
    print("counter", counter)
    if counter > 100:
      break
  print("end")

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
