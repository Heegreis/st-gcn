import os
import json


def get_label_name():
    label_name_path = 'resource/custom_dataset/label_name.txt'
    label_name = []
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]
    return label_name


def loop_dir(train_json_path, dir_path, val_json_path):
    val_json_data = {}

    # 输出所有文件和
    files = os.listdir(dir_path)

    with open(train_json_path, 'r') as load_f:
        train_dict = json.load(load_f)

        for file_name in files:
            file_name_noExtension = file_name.split('.')[0]

            val_json_data[file_name_noExtension] = train_dict[file_name_noExtension]
            del train_dict[file_name_noExtension]

    with open(train_json_path, "w") as dump_f:
        json.dump(train_dict, dump_f)
    
    with open(val_json_path, "w") as dump_f:
        json.dump(val_json_data, dump_f)


if __name__ == "__main__":
    '''
    先拉出來val的pose json，用該程式生成kinetics_val_label.json
    '''
    train_json_path = 'dataset/custom_skeleten_data/noSplitPerson_train_val/kinetics_train_label.json'
    val_dir_path = 'dataset/custom_skeleten_data/noSplitPerson_train_val/kinetics_val'
    val_json_path = 'dataset/custom_skeleten_data/noSplitPerson_train_val/kinetics_val_label.json'
    loop_dir(train_json_path, val_dir_path, val_json_path)