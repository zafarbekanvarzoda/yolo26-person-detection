from ultralytics import YOLO
import cv2


model = YOLO("yolo26n.pt")


def process_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)
        result = results[0]

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            if class_name == "person":

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        out.write(frame)

    cap.release()
    out.release()

    return output_path