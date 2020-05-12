import os
import json

def get_label_name():
    label_name_path = 'resource/custom_dataset/label_name.txt'
    label_name = []
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]
    return label_name
    
def loop_dir(json_dir_path):
    dirs = os.listdir(json_dir_path) # label name
    # 输出所有文件和文件夹
    for label in dirs:
        # print(label)
        class_path = os.path.join(json_dir_path, label)
        files = os.listdir(class_path)
        for file_name in files:
            json_path = os.path.join(class_path, file_name)
            with open(json_path,'r') as load_f:
                load_dict = json.load(load_f)
                load_dict["label"] = label
                load_dict["label_index"] = get_label_name().index(label)

            with open(json_path,"w") as dump_f:
                json.dump(load_dict, dump_f)
                
                

if __name__ == "__main__":
    json_dir_path = 'dataset/custom_skeleten_data/noSplitPerson/poses'
    loop_dir(json_dir_path)