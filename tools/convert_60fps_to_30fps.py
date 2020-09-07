import os
import cv2


def get_video_WH(video_path):
    capture = cv2.VideoCapture(video_path)
    W, H = 0, 0
    while (True):
        # get image
        ret, orig_image = capture.read()
        if not (orig_image is None):
            source_H, source_W, _ = orig_image.shape
            if source_H != 0 and source_W != 0:
                W, H = source_W, source_H
                break
    capture.release()
    return W, H

ori_videos_dir_path = 'dataset/row_video/punch/60fps'
files = os.listdir(ori_videos_dir_path)
for file_name in files:
    video_path = os.path.join(ori_videos_dir_path, file_name)

    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)

    print(video_name)

    # 使用 XVID 編碼
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 建立 VideoWriter 物件，輸出影片至 output.avi
    output_fps = 30.0
    width, height = get_video_WH(video_path)
    out = cv2.VideoWriter('dataset/' + video_name + '.avi', fourcc, output_fps,
                        (1920, 1080))

    frame_index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_index % 1 == 0:
                frame = cv2.resize(frame, (1920, 1080))
                # 寫入影格
                out.write(frame)
                # print(frame_index)

            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #   break
            frame_index += 1
        else:
            break

    # 釋放所有資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()