import os
import json
import csv
import cv2

def mkdirs(dir_path):
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)

def get_video_WH(video_path):
    capture = cv2.VideoCapture(video_path)
    W, H = 0, 0
    while(True):
        # get image
        ret, orig_image = capture.read()
        if not (orig_image is None):
            source_H, source_W, _ = orig_image.shape
            if source_H != 0 and source_W != 0:
                W, H = source_W, source_H
                break
    capture.release()
    return W, H

def loop_dir(ori_videos_dir_path, out_videos_dir_path, out_imgs_dir_path):
    addTime = False
    video_fps = 30
    frame_interval_for_time = 15

    dirs = os.listdir(ori_videos_dir_path)  # label name
    # 输出所有文件和文件夹
    for className in dirs:
        class_path = os.path.join(ori_videos_dir_path, className)
        files = os.listdir(class_path)
        for file_name in files:
            video_path = os.path.join(class_path, file_name)
            video_name = video_path.split('/')[-1].split('.')[0]

            width, height = get_video_WH(video_path)
            video_capture = cv2.VideoCapture(video_path)
            mkdirs(os.path.join(out_videos_dir_path, className))
            video_output_path = os.path.join(out_videos_dir_path, className, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            print(video_name)
            out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))

            frame_index = 0

            while(True):
                frame_index += 1

                # get image
                ret, orig_image = video_capture.read()
                if orig_image is None:
                    break
                
                # 在影片中標示時間
                if addTime:
                    time_screen = (frame_index // frame_interval_for_time) * 0.5
                    cv2.putText(orig_image,
                                str(time_screen) + ' sec', (50, 70), 0, 1.8, (0, 0, 0), 5)
                    cv2.putText(orig_image,
                                str(time_screen) + ' sec', (50, 70), 0, 1.8, (0, 255, 0),
                                3)

                cv2.putText(orig_image, str(frame_index), (20, 30), 0, 1,
                            (0, 255, 0), 2)
                out.write(orig_image)

                mkdirs(os.path.join(out_imgs_dir_path, className, video_name))
                img_path = os.path.join(out_imgs_dir_path, className, video_name, video_name + '_' + str(frame_index).zfill(3) + '.jpg')
                cv2.imwrite(img_path, orig_image)


            video_capture.release()
            out.release()

def loop_single_dir(ori_videos_dir_path, out_videos_dir_path, out_imgs_dir_path):
    addTime = True
    video_fps = 30
    frame_interval_for_time = 15

    files = os.listdir(ori_videos_dir_path)  # label name
    
    for file_name in files:
        video_path = os.path.join(ori_videos_dir_path, file_name)
        video_name = video_path.split('/')[-1].split('.')[0]

        width, height = get_video_WH(video_path)
        video_capture = cv2.VideoCapture(video_path)
        video_output_path = os.path.join(out_videos_dir_path, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(video_name)
        out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))

        frame_index = 0

        while(True):
            frame_index += 1

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            
            # 在影片中標示時間
            if addTime:
                time_screen = (frame_index // frame_interval_for_time) * 0.5
                cv2.putText(orig_image,
                            str(time_screen) + ' sec', (50, 70), 0, 1.8, (0, 0, 0), 5)
                cv2.putText(orig_image,
                            str(time_screen) + ' sec', (50, 70), 0, 1.8, (0, 255, 0),
                            3)

            cv2.putText(orig_image, str(frame_index), (20, 30), 0, 1,
                        (0, 255, 0), 2)
            out.write(orig_image)

            mkdirs(os.path.join(out_imgs_dir_path, video_name))
            img_path = os.path.join(out_imgs_dir_path, video_name, video_name + '_' + str(frame_index).zfill(3) + '.jpg')
            cv2.imwrite(img_path, orig_image)


        video_capture.release()
        out.release()


if __name__ == "__main__":
    # ori_videos_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/videos'
    # out_videos_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/videos_with_frame_index'
    # out_imgs_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/frame_imgs'
    
    # ori_videos_dir_path = 'dataset/test/custom_skeleten_data/forSplitPerson_with_ID/videos'
    # out_videos_dir_path = 'dataset/test/custom_skeleten_data/forSplitPerson_with_ID/videos_with_frame_index'
    # out_imgs_dir_path = 'dataset/test/custom_skeleten_data/forSplitPerson_with_ID/frame_imgs'
    
    # ori_videos_dir_path = 'data/mydata/test_output_custom_demo/0601_nolabel'
    # out_videos_dir_path = 'data/mydata/test_output_custom_demo/0601_nolabel_with_frame_index'
    # out_imgs_dir_path = 'data/mydata/test_output_custom_demo/0601_nolabel_frame_imgs'

    # ori_videos_dir_path = 'dataset/custom_skeleten_data/extra_with_ID/videos'
    # out_videos_dir_path = 'dataset/custom_skeleten_data/extra_with_ID/videos_with_frame_index'
    # out_imgs_dir_path = 'dataset/custom_skeleten_data/extra_with_ID/frame_imgs'

    ori_videos_dir_path = 'data/mydata/test_output_custom_demo/newK_video/正式'
    out_videos_dir_path = 'data/mydata/test_output_custom_demo/newK_video_with_frame_index/正式'
    out_imgs_dir_path = 'data/mydata/test_output_custom_demo/newK_video_frame_imgs/正式'

    mkdirs(out_videos_dir_path)
    mkdirs(out_imgs_dir_path)

    # loop_dir(ori_videos_dir_path, out_videos_dir_path, out_imgs_dir_path)

    loop_single_dir(ori_videos_dir_path, out_videos_dir_path, out_imgs_dir_path)