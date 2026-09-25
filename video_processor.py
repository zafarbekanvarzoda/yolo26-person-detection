from ultralytics import YOLO
import cv2


model = YOLO("yolo26n.pt")


def process_video(input_path, output_path):

    print("PROCESS VIDEO STARTED")

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

    print("VideoWriter opened:", out.isOpened())

    timeline = []

    max_people = 0

    frame_number = 0
    last_second = -1

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model(frame)

        result = results[0]

        people_count = 0

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            class_name = model.names[class_id]

            if class_name == "person":

                people_count += 1

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

        if people_count > max_people:
            max_people = people_count

        current_time = (frame_number - 1) / fps
        current_second = int(current_time)

        if current_second != last_second:

            timeline.append({
                "time": current_second,
                "people": people_count
            })

            last_second = current_second

        out.write(frame)

    cap.release()
    out.release()

    return {
        "video": output_path,
        "max_people": max_people,
        "timeline": timeline
    }
