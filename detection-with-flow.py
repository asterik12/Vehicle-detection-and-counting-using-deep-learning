#detection with lucas kanade optical flow 

import cv2
import numpy as np
import datetime
yolo = cv2.dnn.readNet("./data/yolov3.weights", "./data/yolov3.cfg")
classes = []

with open("./data/coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

colorwhite = (242, 242, 242)
datecolor=(113,149,12)
font_scale = 1.2
font = cv2.FONT_HERSHEY_COMPLEX

# set the rectangle background to white
rectangle_bgr = (128,0,128)
count=0
total_count=0
# #Loading Images
#name = "market.jpg"
#img = cv2.imread(name)


cap = cv2.VideoCapture("./src/video/traffic-footage3.mp4")
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
#result = cv2.VideoWriter('Result-23.avi',cv2.VideoWriter_fourcc(*'MJPG'),25, size) 



#optical-flow
feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 3, 
                       blockSize = 3 ) 
  
# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              1, 0.03)) 
  
# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 
  
# Take first frame and find corners in it 
ret, old_frame = cap.read() 
old_gray = cv2.cvtColor(old_frame, 
                        cv2.COLOR_BGR2GRAY) 
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, 
                             **feature_params) 
  
# Create a mask image for drawing purposes 
mask = np.zeros_like(old_frame) 

while True:
    _,img = cap.read()
    height, width, _ = img.shape
    img[0:50,0:width]=[242,242,242]
    cv2.putText(img,'VEHICLE COUNT:',(10,38),cv2.FONT_HERSHEY_SIMPLEX,1.2,(43,65,149),2)
    
    cv2.line(img,(260,height-110),(width-100,height-110),(0,255,255),4)
    
    
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
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
    
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4) #set confidence value
    
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],1))
            #couting vehicles
            get_center_box=int(y+h/2)
            get_line_dist=height-110
            if label=='car' or label=='truck' or label=='motorbike' or label=='bus':
            	if((get_center_box<get_line_dist+4 and get_center_box>get_line_dist-4)):
            	    count=count+1 
            	    cv2.line(img,(260,height-110),(width-100,height-110),(0,0,255),2)
            	
            if label=='car':
                color=(135,110,8)
            elif label=='bus':
                color=(9,90,144)
            elif label=='truck':
                color=(113,149,12)	
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=4)[0]
            box_coords = ((x, y), (x + text_width , y - text_height ))
            cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
            cv2.putText(img, label+" "+confidence, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorwhite, 1)
            cv2.putText(img,str(count),(340,38),cv2.FONT_HERSHEY_SIMPLEX,1.5,(43,65,149),4)
            
            dt = str(datetime.datetime.now())
            img=cv2.putText(img,dt,(900,38),font,1,datecolor,2,cv2.LINE_8)
            
    # Create some random colors 
    color = np.random.randint(0, 255, (100, 3)) 
    #optical-flow
    frame_gray = cv2.cvtColor(img, 
                              cv2.COLOR_BGR2GRAY) 
  
    # calculate optical flow 
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
                                           frame_gray, 
                                           p0, None, 
                                           **lk_params) 
  
    # Select good points 
    good_new = p1[st == 1] 
    good_old = p0[st == 1] 
  
    # draw the tracks 
    for i, (new, old) in enumerate(zip(good_new,  
                                       good_old)): 
        a, b = new.ravel() 
        c, d = old.ravel() 
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1) 
        img = cv2.circle(img, (a, b), 5, color[i].tolist(), -1) 
          


    #get video fps
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        cv2.putText(img,"FPS: "+format(fps),(500,38),cv2.FONT_HERSHEY_SIMPLEX,1.5,(43,65,149),4)
        
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img,"FPS: "+format(fps),(500,38),cv2.FONT_HERSHEY_SIMPLEX,1.3,(43,65,149),3)
       
       
    img = cv2.add(img, mask) 
    
    cv2.imshow("Image", img)
    #result.write(img) 
    #print('Number of objects in the image is ',count)
    #cv2.imwrite("output.mp4",img)
    key = cv2.waitKey(1)
    if key==27:
        break
    
    # Updating Previous frame and points      
    old_gray = frame_gray.copy() 
    p0 = good_new.reshape(-1, 1, 2) 
cap.release()
cv2.destroyAllWindows()
