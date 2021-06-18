# %%
import numpy as np
import time
import cv2
import streamlit as st
import os

# %%
st.title("Poker Card Screener")
upload_image = st.sidebar.file_uploader("Upload an image here:")

threshold = st.sidebar.slider("Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
model = st.sidebar.selectbox('Model',['model 1', 'model 2'])

model_path = '../backup/yolov4-obj_best.weights'
LABELS_FILE='../yolov4/obj-2.names'

if model == 'model 1':
    model_path = '../backup/yolov4-obj-rate-0001_best.weights'
    LABELS_FILE='../yolov4/obj.names'

if upload_image is not None:
    # OUTPUT_FILE='predicted.jpg'
    CONFIG_FILE='../yolov4/yolov4-obj.cfg'
    WEIGHTS_FILE=model_path
    CONFIDENCE_THRESHOLD=threshold

    LABELS = open(LABELS_FILE).read().strip().split("\n")

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    image = ''
    
    with open(upload_image.name,'wb') as f:
        f.write(upload_image.read())
        image = cv2.imread(upload_image.name)
        os.remove(upload_image.name)    
    
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    center_boxes = []
    frame = [{
        "frame_id": 1,
        "file_name": upload_image.name,
         "objects": []
    }]
    # %%
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                center_boxes.append([centerX, centerY, width, height])
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

    # %%
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            
            if image.shape[1] > 1500:
                text_size = 2
            else:
                text_size = 0.5
                
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                text_size, color, 3)
            
            frame[0]["objects"].append({
                "class_id": classIDs[i],
                "name": LABELS[classIDs[i]],
                "relative_coordinates": {
                    "center_x": center_boxes[i][0],
                    "center_y":center_boxes[i][1], "width":center_boxes[i][2], "height":center_boxes[i][3]
                },
                "confidence": confidences[i]
            })

    # show the output image
    # cv2.imwrite(OUTPUT_FILE, image)
    
    st.image(image, channels="BGR")
    # %%
    
    def card_parser(x):
        dealer = []
        player = []
        
        for i in x[0]['objects']:
            if i["relative_coordinates"]["center_y"] < 0.5 * H:
                dealer.append(i['name'])
            else:
                player.append(i['name'])
        
        return list(set(dealer)), list(set(player))
    
    def card_point(card):
        if card[:-1] in ['10','J','Q','K']:
            return 10
        elif card[:-1] == 'A':
            return 1
        else:
            return int(card[:-1])
        
    strategy_table = [['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','D','D','D','D','H','H','H','H'],
                ['D','D','D','D','D','D','D','D','D','D'],
                ['H','D','D','D','D','D','D','D','D','H'],
                ['H','H','H','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S']
                ]
    
    strategy_table_A = [['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','H','H','H','H','H','H'],
                ['H','H','H','H','D','D','H','H','H','H'],
                ['H','H','H','H','D','D','H','H','H','H'],
                ['H','H','H','D','D','D','H','H','H','H'],
                ['H','H','H','D','D','D','H','H','H','H'],
                ['H','H','D','D','D','D','H','H','H','H'],
                ['H','D','D','D','D','D','S','S','H','H'],
                ['S','S','S','S','S','D','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['H','H','H','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['H','S','S','S','S','S','H','H','H','H'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S'],
                ['S','S','S','S','S','S','S','S','S','S']
                ]
    real_meaning = {'H':'Hit','D':'Double your bet','S':'Stand'}

    def strategy(dealer, player):
        if len(dealer) > 0:
            dealer_point = card_point(dealer[0])
            player_point = sum([card_point(i) for i in player])
            if player_point > 21:
                return 'Busted'
            elif 1 in [card_point(i) for i in player]:
                return real_meaning[strategy_table_A[player_point-1][dealer_point-1]]
            else:
                return real_meaning[strategy_table[player_point-1][dealer_point-1]]
        else:
            return "Can't recognize dealer's card"
            
    st.write(f"Dealer's card: {card_parser(frame)[0]}")
    st.write(f"Player's card: {card_parser(frame)[1]}")
    st.write(strategy(card_parser(frame)[0],card_parser(frame)[1]))