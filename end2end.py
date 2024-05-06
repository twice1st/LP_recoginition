import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml



def draw_detections(img, res_detect, classes, text):
  for ele in res_detect:
  # Extract the coordinates of the bounding box
    x1, y1, w, h = ele[0]
    # Retrieve the color for the class ID
    # color = np.random.uniform(0, 255, size=(len(classes), 3))
    color=(255, 0, 0)
    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1)

    # Create the label text with class name and score
    label = f"{classes[ele[2]]}: {ele[1]:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    if text:
      cv2.rectangle(
          img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
      )
      cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
  return img




def preprocess(img, input_width, input_height):
  """
  Preprocesses the input image before performing inference.

  Returns:
      image_data: Preprocessed image data ready for inference.
  """
  # Convert the image color space from BGR to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Resize the image to match the input shape
  img = cv2.resize(img, (input_width, input_height))

  # Normalize the image data by dividing it by 255.0
  image_data = np.array(img) / 255.0

  # # Transpose the image to have the channel dimension as the first dimension
  image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

  # Expand the dimensions of the image data to match the expected input shape
  image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

  # Return the preprocessed image data
  return image_data




def postprocess(input_image, output, input_width, input_height, classes, text=True, confidence_thres=0.5, iou_thres=0.5):
  # Transpose and squeeze the output to match the expected shape
  outputs = np.transpose(np.squeeze(output[0]))

  # Get the number of rows in the outputs array
  rows = outputs.shape[0]

  # Lists to store the bounding boxes, scores, and class IDs of the detections
  boxes = []
  scores = []
  class_ids = []

  # Calculate the scaling factors for the bounding box coordinates
  img_height, img_width = input_image.shape[:2]
  x_factor = img_width / input_width
  y_factor = img_height / input_height

  # Iterate over each row in the outputs array
  for i in range(rows):
      # Extract the class scores from the current row
      classes_scores = outputs[i][4:]

      # Find the maximum score among the class scores
      max_score = np.amax(classes_scores)

      # If the maximum score is above the confidence threshold
      if max_score >= confidence_thres:
          # Get the class ID with the highest score
          class_id = np.argmax(classes_scores)

          # Extract the bounding box coordinates from the current row
          x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

          # Calculate the scaled coordinates of the bounding box
          left = int((x - w / 2) * x_factor)
          top = int((y - h / 2) * y_factor)
          width = int(w * x_factor)
          height = int(h * y_factor)

          # Add the class ID, score, and box coordinates to the respective lists
          class_ids.append(class_id)
          scores.append(max_score)
          boxes.append([left, top, width, height])

  # Apply non-maximum suppression to filter out overlapping bounding boxes
  indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
  lst_results = []
  # Iterate over the selected indices after non-maximum suppression
  for i in indices:
      # Get the box, score, and class ID corresponding to the index
      box = boxes[i]
      score = scores[i]
      class_id = class_ids[i]
      lst_results.append([box, score, class_id])
      # Draw the detection on the input image
      # draw_detections(input_image, box, score, class_id, classes, text)

  # Return the modified input image
  return lst_results, y_factor



def inference_detect_onnxModel(image):
  session = ort.InferenceSession(r'D:\LP_recognition\train\train_detect\weights\best_detect.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  model_inputs = session.get_inputs()

  # Store the shape of the input for later use
  input_shape = model_inputs[0].shape
  input_width = input_shape[3]
  input_height = input_shape[2]
  # Preprocess the image data
  start = time.time()
  img_data = preprocess(image, input_width, input_height)
  # Run inference using the preprocessed image data
  outputs = session.run(None, {model_inputs[0].name: img_data})
  end = time.time()
  classes = yaml_load(check_yaml(r"D:\LP_recognition\yaml\detect.yaml"))["names"]
  # Perform post-processing on the outputs to obtain output image.
  res_detect,_ = postprocess(image, outputs, input_width, input_height, classes)

  img_draw = draw_detections(image.copy(), res_detect, classes, text = True)
  # cv2_imshow(img)
  x,y,w,h = res_detect[0][0]
  return image[y:y+h, x:x+w], img_draw, end-start


def inference_recog_onnxModel(image):
  session = ort.InferenceSession(r'D:\LP_recognition\train\train_recog\weights\best_recog.onnx', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  model_inputs = session.get_inputs()

  # Store the shape of the input for later use
  input_shape = model_inputs[0].shape
  input_width = input_shape[3]
  input_height = input_shape[2]
  # Preprocess the image data
  # image = cv2.imread(input_image)
  h_lp, _, _ = image.shape
#   print(h_lp)
  start = time.time()
  img_data = preprocess(image, input_width, input_height)
  # Run inference using the preprocessed image data
  outputs = session.run(None, {model_inputs[0].name: img_data})
  end = time.time()
  classes = yaml_load(check_yaml(r"D:\LP_recognition\yaml\recog.yaml"))["names"]
  # Perform post-processing on the outputs to obtain output image.
  res_recog, y_factor =  postprocess(image, outputs, input_width, input_height, classes, text=False)

  img_draw = draw_detections(image.copy(), res_recog, classes, text = False)

  conf=0.5
  iou=0.7
  count_num_of_loop = 0
  while True:
    count_num_of_loop += 1
    if count_num_of_loop > 10:
      print('error !!!')
      return
    print('number of loop: ', count_num_of_loop)

    if len(res_recog) < 8:
      res_recog,_ = postprocess(image, outputs, input_width, input_height, classes, text=False, conf_thres=conf-0.05, iou_thres=iou+0.05)

    if len(res_recog) > 9:
      res_recog = sorted(res_recog, key=lambda x:x[1])[:9]

    if len(res_recog) > 7 & len(res_recog) < 10:
      break
  # return res_recog
  lst_char_line_1 = []
  lst_char_line_2 = []
  for keyword in res_recog:
    x, y, w, h = keyword[0]
    # print(x, y, w, h)
    label = keyword[2]
    conf = keyword[1]
    # print(conf, label)
    label = classes[int(label)]
    tmp = [x, y, w, h, round(float(conf),4), label]
    # lst_char_line_1.append(image.copy()[y:y+h, x:x+w])
    if 0 < (y+h) / 2 < h_lp*y_factor / 2:
      lst_char_line_1.append(tmp)
    else:
      lst_char_line_2.append(tmp)
  lst_char_line_1 = sorted(lst_char_line_1, key=lambda x:x[0])
  lst_char_line_2 = sorted(lst_char_line_2, key=lambda x:x[0])

  res_of_label = [*[x[5] for x in lst_char_line_1],"  ",*[y[5] for y in lst_char_line_2]]
  # res_of_label = ['5', 'D', '4', '1', '6', '6', '4', '8', '0']

  ## Heurictic rule
  char2num = {'A':'4', 'B': '8', 'D':'0', 'G':'6', 'S':'5', 'Z':'7'}
  num2char = {'0':'D', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'Z', '8':'B'}
  for idx, ele in enumerate(res_of_label):
    if idx == 2:
      if ele in num2char.keys():
        res_of_label[idx] = num2char[ele]
        print(res_of_label)
    else:
      if ele in char2num.keys():
        res_of_label[idx] = char2num[ele]
  res_of_label = ''.join(res_of_label)
  return res_of_label, img_draw, end-start

def end2end(image):
  inp_recog, img_draw_detect, t1 = inference_detect_onnxModel(image)
  res_lp, image_draw_recog, t2 = inference_recog_onnxModel(inp_recog)
  return t1+t2, res_lp, img_draw_detect, inp_recog, image_draw_recog
#   cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#   cv2.imshow("Output", image_draw_recog)
#   cv2.waitKey(0)
# ima = Image.open(r'D:\LP_recognition\GreenParking_Detect\data_Detect\0.jpg')
# ima = np.array(ima)
# end2end(ima)