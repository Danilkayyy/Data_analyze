import cv2
from datetime import datetime

from solution import Camera


def process_video():
    # необходимо указать корректный id камеры
    camera = Camera("/dev/video2")
    camera.process()
    csv_filename = "test.npy"
    camera.save_file(csv_filename)
    print(f"saved in {csv_filename}")


def write_video():
    cap = cv2.VideoCapture("/dev/video2")
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    outvideo = cv2.VideoWriter()
    outvideo.open(f'{datetime.now().time()}.mp4', fourcc, 30, sz, True)

    while True:
        _, frame = cap.read()

        outvideo.write(frame)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    outvideo.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
    # write_video()