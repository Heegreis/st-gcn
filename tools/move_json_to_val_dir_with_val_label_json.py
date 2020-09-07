import os
import json
import shutil

def loop_dir(train_dir_path, val_dir_path, val_json_path):

    with open(val_json_path, 'r') as load_f:
        load_dict = json.load(load_f)

        for key in load_dict:
            print(key)
            shutil.move(os.path.join(train_dir_path, key + '.json'), os.path.join(val_dir_path, key + '.json'))



if __name__ == "__main__":
    '''
    根據kinetics_val_label.json將pose json檔移到val資料夾
    '''
    # train_dir_path = 'dataset/custom_skeleten_data/splitPerson_train_val/kinetics_train'
    # val_dir_path = 'dataset/custom_skeleten_data/splitPerson_train_val/kinetics_val'
    # val_json_path = 'dataset/custom_skeleten_data/splitPerson_train_val/kinetics_val_label.json'

    train_dir_path = 'dataset/custom_skeleten_data/splitPersonEssenceMoreNature_train_val/kinetics_train'
    val_dir_path = 'dataset/custom_skeleten_data/splitPersonEssenceMoreNature_train_val/kinetics_val'
    val_json_path = 'dataset/custom_skeleten_data/splitPersonEssenceMoreNature_train_val/kinetics_val_label.json'

    loop_dir(train_dir_path, val_dir_path, val_json_path)