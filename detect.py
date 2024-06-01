import os
import torch
import cv2

# Verify if the image file exists
image_path = '/Users/yacoubosman/Desktop/NeuraVision/yolov5/data/images/photo.jpg'
if not os.path.exists(image_path):
    print(f"Error: The image file {image_path} does not exist.")
else:
    # Download and load the YOLOv5n model from github
    print("Loading model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    print("Model loaded successfully.")

    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image from path: {image_path}")
    else:
        img = cv2.resize(img, (1000, 650))

        # Perform detection on the image
        result = model(img)
        print('Result: ', result)

        # Convert detected result to pandas DataFrame
        data_frame = result.pandas().xyxy[0]
        print('Data Frame:')
        print(data_frame)

        # Get indexes of all the rows
        indexes = data_frame.index
        for index in indexes:
            # Find the coordinates of the top left corner of the bounding box
            x1 = int(data_frame['xmin'][index])
            y1 = int(data_frame['ymin'][index])
            # Find the coordinates of the bottom right corner of the bounding box
            x2 = int(data_frame['xmax'][index])
            y2 = int(data_frame['ymax'][index])

            # Find label name
            label = data_frame['name'][index]
            # Find confidence score of the model
            conf = data_frame['confidence'][index]
            text = label + ' ' + str(conf.round(decimals=2))

            # Draw the bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Display the image with detections
        cv2.imshow('IMAGE', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
