import os
import json


def get_label_name():
    label_name_path = 'resource/custom_dataset/label_name.txt'
    label_name = []
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]
    return label_name


def loop_dir(json_dir_path, out_label_json_dir_path):
    out_label_json_path = os.path.join(out_label_json_dir_path, 'dataset_label.json')
    
    output_json_data = {}

    dirs = os.listdir(json_dir_path)  # label name
    # 输出所有文件和文件夹
    for className in dirs:
        # print(label)
        class_path = os.path.join(json_dir_path, className)
        files = os.listdir(class_path)
        for file_name in files:
            file_name_noExtension = file_name.split('.')[0]

            json_path = os.path.join(class_path, file_name)
            with open(json_path, 'r') as load_f:
                load_dict = json.load(load_f)
                label = load_dict['label']
                label_index = load_dict['label_index']
            
            content_data = {
                "has_skeleton": True,
                "label": label,
                "label_index": label_index
            }
            output_json_data[file_name_noExtension] = content_data

    with open(out_label_json_path, "w") as dump_f:
        json.dump(output_json_data, dump_f)


if __name__ == "__main__":
    # json_dir_path = 'dataset/test/custom_skeleten_data/poses'
    # out_label_json_dir_path = 'dataset/test/custom_skeleten_data'

    # json_dir_path = 'dataset/custom_skeleten_data/noSplitPerson/poses'
    # out_label_json_dir_path = 'dataset/custom_skeleten_data/noSplitPerson'

    json_dir_path = 'dataset/custom_skeleten_data/splitPerson/poses'
    out_label_json_dir_path = 'dataset/custom_skeleten_data/splitPerson'

    loop_dir(json_dir_path, out_label_json_dir_path)