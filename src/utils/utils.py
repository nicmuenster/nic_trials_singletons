import os

def condense_checkpoints(folder_name, checkpoint_name):
    file_list= os.listdir(folder_name)
    ckpt_list = []
    for file in file_list:
        if file.endswith(".ckpt") and file != checkpoint_name:
            ckpt_list.append(file)
    if ckpt_list:
        os.rename(folder_name + ckpt_list.pop(), folder_name + checkpoint_name + ".ckpt")
    for ckpt in ckpt_list:
        os.remove(folder_name + ckpt)