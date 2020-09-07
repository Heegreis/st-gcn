import os
import json
import csv
import copy


def mkdirs(out_json_dir):
    if os.path.isdir(out_json_dir) == False:
        os.makedirs(out_json_dir)

def loop_dir_for_get_all_file_name(json_dir_path, out_name_list_dir_path):
    dirs = os.listdir(json_dir_path)  # label name
    # 输出所有文件和文件夹
    for className in dirs:
        # 開啟輸出的 CSV 檔案
        csv_file_path = os.path.join(out_name_list_dir_path, className + '.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            
            class_path = os.path.join(json_dir_path, className)
            files = os.listdir(class_path)
            files.sort()
            for file_name in files:
                file_name_noExtension = file_name.split('.')[0]

                # 寫入一列資料
                writer.writerow([file_name_noExtension])

def loop_dir_onePerson(json_dir_path, name_list_dir_path, out_json_dir_path):
    dirs = os.listdir(json_dir_path)  # label name
    # 输出所有文件和文件夹
    for className in dirs:
        # print(label)
        class_path = os.path.join(json_dir_path, className)

        csv_path = os.path.join(name_list_dir_path, className + '.csv')
        if not os.path.isfile(csv_path):
            continue
        # 開啟 CSV 檔案
        with open(csv_path, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)

            # 以迴圈輸出每一列
            for row in rows:
                file_name_noExtension = row[0]
                keep_id = row[1]
                print(file_name_noExtension)

                json_path = os.path.join(class_path, file_name_noExtension + '.json')
                with open(json_path, 'r') as load_f:
                    load_dict = json.load(load_f)
                for i, frame_data in enumerate(load_dict['data']):
                    del_index = []
                    for j, skeleton in enumerate(frame_data['skeleton']):
                        if skeleton['id'] != int(keep_id):
                            del_index.append(j)
                        del skeleton['id']
                        frame_data['skeleton'][j] = skeleton
                    # print(del_index)
                    for index in sorted(del_index, reverse=True):
                        del frame_data['skeleton'][index]
                    # print(frame_data['skeleton'])
                    load_dict['data'][i] = frame_data
                
                out_json_dir = os.path.join(out_json_dir_path, className)
                mkdirs(out_json_dir)
                out_json_path = os.path.join(out_json_dir, file_name_noExtension + '.json')
                with open(out_json_path, "w") as dump_f:
                    json.dump(load_dict, dump_f)

def loop_dir_onePerson_essence(json_dir_path, name_list_dir_path, out_json_dir_path):
    dirs = os.listdir(json_dir_path)  # label name
    # 输出所有文件和文件夹
    for className in dirs:
        # print(label)
        class_path = os.path.join(json_dir_path, className)

        csv_path = os.path.join(name_list_dir_path, className + '.csv')
        if not os.path.isfile(csv_path):
            continue
        # 開啟 CSV 檔案
        with open(csv_path, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)

            # 以迴圈輸出每一列
            for row in rows:
                file_name_noExtension = row[0]
                keep_id = row[1]
                action_start_frame_index = row[2]
                action_end_frame_index = row[3]
                print(file_name_noExtension)

                json_path = os.path.join(class_path, file_name_noExtension + '.json')
                with open(json_path, 'r') as load_f:
                    load_dict = json.load(load_f)
                for i, frame_data in enumerate(load_dict['data']):
                    del_index = []
                    for j, skeleton in enumerate(frame_data['skeleton']):
                        if skeleton['id'] != int(keep_id):
                            del_index.append(j)
                        del skeleton['id']
                        frame_data['skeleton'][j] = skeleton
                    # print(del_index)
                    for index in sorted(del_index, reverse=True):
                        del frame_data['skeleton'][index]
                    # print(frame_data['skeleton'])
                    load_dict['data'][i] = frame_data
                
                before_dict = copy.deepcopy(load_dict)
                essence_dict = copy.deepcopy(load_dict)
                after_dict = copy.deepcopy(load_dict)
                for i, frame_data in enumerate(before_dict['data']):
                    if frame_data['frame_index'] >= int(action_start_frame_index):
                        frame_data['skeleton'].clear()
                for i, frame_data in enumerate(essence_dict['data']):
                    if frame_data['frame_index'] < int(action_start_frame_index) or frame_data['frame_index'] > int(action_end_frame_index):
                        frame_data['skeleton'].clear()
                for i, frame_data in enumerate(after_dict['data']):
                    if frame_data['frame_index'] <= int(action_end_frame_index):
                        frame_data['skeleton'].clear()
                
                out_before_json_dir = os.path.join(out_json_dir_path, className, className + '_before')
                out_essence_json_dir = os.path.join(out_json_dir_path, className, className + '_essence')
                out_after_json_dir = os.path.join(out_json_dir_path, className, className + '_after')
                mkdirs(out_before_json_dir)
                mkdirs(out_essence_json_dir)
                mkdirs(out_after_json_dir)
                out_before_json_path = os.path.join(out_before_json_dir, file_name_noExtension + '_before.json')
                out_essence_json_path = os.path.join(out_essence_json_dir, file_name_noExtension + '_essence.json')
                out_after_json_path = os.path.join(out_after_json_dir, file_name_noExtension + '_after.json')
                with open(out_before_json_path, "w") as dump_f:
                    json.dump(before_dict, dump_f)
                with open(out_essence_json_path, "w") as dump_f:
                    json.dump(essence_dict, dump_f)
                with open(out_after_json_path, "w") as dump_f:
                    json.dump(after_dict, dump_f)

if __name__ == "__main__":
    # json_dir_path = 'dataset/test/custom_skeleten_data/poses'
    # out_json_dir_path = 'dataset/test/custom_skeleten_data/new_poses'
    # out_name_list_dir_path = 'dataset/test/custom_skeleten_data'

    # json_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/poses'
    # out_json_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/new_poses'
    # out_name_list_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID'

    # json_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/poses'
    # out_json_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID/new_poses_essence'
    # out_name_list_dir_path = 'dataset/custom_skeleten_data/forSplitPerson_with_ID'

    json_dir_path = 'dataset/custom_skeleten_data/extra_with_ID/poses'
    out_json_dir_path = 'dataset/custom_skeleten_data/extra_with_ID/new_poses_essence'
    out_name_list_dir_path = 'dataset/custom_skeleten_data/extra_with_ID'

    loop_dir_onePerson_essence(json_dir_path, out_name_list_dir_path, out_json_dir_path)
    # loop_dir_for_get_all_file_name(json_dir_path, out_name_list_dir_path)