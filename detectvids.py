import os
import torch
import cv2

# Path to the video file
video_path = '/Users/yacoubosman/Desktop/NeuraVision/yolov5/data/videos/Movie.mp4'

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The video file {video_path} does not exist.")
else:
    # Load the YOLOv5n model from github
    print("Loading model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    print("Model loaded successfully.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
    else:
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (1000, 650))

            # Perform detection on the frame
            result = model(frame)
            data_frame = result.pandas().xyxy[0]

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

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # Display the frame with detections
            cv2.imshow('VIDEO', frame)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close display windows
        cap.release()
        cv2.destroyAllWindows()
