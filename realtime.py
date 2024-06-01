import torch
import cv2

# Replace with your Manything stream URL
stream_url = 'https://app.manything.com/cameras/grid'

# Load the YOLOv5 model
print("Loading model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
print("Model loaded successfully.")

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"Error: Unable to open video stream: {stream_url}")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1000, 650))
        result = model(frame)
        data_frame = result.pandas().xyxy[0]

        for index in data_frame.index:
            x1 = int(data_frame['xmin'][index])
            y1 = int(data_frame['ymin'][index])
            x2 = int(data_frame['xmax'][index])
            y2 = int(data_frame['ymax'][index])
            label = data_frame['name'][index]
            conf = data_frame['confidence'][index]
            text = label + ' ' + str(conf.round(decimals=2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        cv2.imshow('VIDEO', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
