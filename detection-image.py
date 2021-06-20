import cv2
import numpy as np

yolo = cv2.dnn.readNet("./data/yolov3.weights", "./data/yolov3.cfg")
classes = []

with open("./data/coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

colorwhite = (242,242,242)
colorPurple = (128,0,128)

# #Loading Images

img = cv2.imread("./src/image/source-cars.png")
height, width, channels = img.shape

font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX

# set the rectangle background to white
rectangle_bgr = (128,0,128)
count=0
total_count=0

#adding top line
img[0:40,0:width]=[242, 242, 242]
cv2.putText(img,'Vehicles Counted:',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(43,65,149),2)

# # Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

yolo.setInput(blob)
outputs = yolo.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4) #set confidence value
COLORS = np.random.uniform(0,255,size=(len(classes),2))

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],1))
        if label=='car' or label=='bus' or label=='truck':
            count=count+1
        total_count=total_count+1
        color = COLORS[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]
        box_coords = ((x, y), (x + text_width, y - text_height))
        cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
        cv2.putText(img, "{} {:.2f}".format(label, float(confidence)), (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorwhite,1)
	

cv2.putText(img,str(count),(300,38),cv2.FONT_HERSHEY_SIMPLEX,1.5,(43,65,149),2)
        
cv2.imshow("market", img)
print('Number of vehicles in the image is ',count)
print('\nNumber of objects in the image is ',total_count)
#cv2.imwrite("output.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
