import argparse
import json
import os
# "/mnt/c/Users/nicmu/workspace/nic_paper_singletons/configs/inventory_builder.json"
# "/mnt/c/Users/nicmu/Desktop/testordner"

if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('inventory_builder')
    parser.add_argument('--basic_config_path', default='./config_files/', help='the basic config needed for buiding')
    parser.add_argument('--experiment_name', default='chestxray',
                        help='name of the subfolder structure')

    args = parser.parse_args()

    # read config
    with open(args.basic_config_path, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)
    config["data_set"] = args.experiment_name
    #config["experiment_root"] = "/mnt/c/Users/nicmu/Desktop/testordner/"
    config["image_root"] = config["experiment_root"] + args.experiment_name + "/images/"
    config["csv_train"] = config["experiment_root"] + args.experiment_name + config["csv_train"]
    config["csv_val"] = config["experiment_root"] + args.experiment_name + config["csv_val"]
    config["csv_test"] = config["experiment_root"] + args.experiment_name + config["csv_test"]
    config["csv_train_singles"] = config["experiment_root"] + args.experiment_name + config["csv_train_singles"]
    config["csv_val_singles"] = config["experiment_root"] + args.experiment_name + config["csv_val_singles"]
    config["csv_test_singles"] = config["experiment_root"] + args.experiment_name + config["csv_test_singles"]
    path0 = config["experiment_root"] + args.experiment_name
    if not (os.path.exists(path0)):
        os.mkdir(path0)
    path0 = path0 + "/trials"
    if not (os.path.exists(path0)):
        os.mkdir(path0)
    with open(path0 + "/test_config.json", "w") as config_out:
        json.dump(config, config_out)
    for leave_in in range(6):
        current_num = leave_in * 20
        path1 = path0 + "/singles" + str(current_num)
        if not (os.path.exists(path1)):
            os.mkdir(path1)
        for put_at_end in range(6):
            current_inner_num = put_at_end * 20
            path2 = path1 + "/end" + str(current_inner_num)
            if not (os.path.exists(path2)):
                os.mkdir(path2)
            for fold in range(1):
                intermediate_config = config.copy()
                path3 = path2 + "/fold" + str(fold+1)
                if not (os.path.exists(path3)):
                    os.mkdir(path3)
                if not (os.path.exists(path3 + config["log_path"])):
                    os.mkdir(path3 + config["log_path"])

                '''if not (os.path.exists(path3 + config["best_model_path"])):
                    os.mkdir(path3 + config["best_model_path"])
                if not (os.path.exists(path3 + config["checkpoint_path"])):
                    os.mkdir(path3 + config["checkpoint_path"])'''
                intermediate_config["checkpoint_folder"] = path3 + "/"
                intermediate_config["checkpoint_path"] = path3 + "/" + args.experiment_name + \
                                                         str(current_num) + "_" + str(current_inner_num) + \
                                                         "_intermediate.ckpt"
                intermediate_config["checkpoint_name"] = args.experiment_name + \
                                                         str(current_num) + "_" + str(current_inner_num) + \
                                                         "_intermediate"
                intermediate_config["best_model_path"] = path3 + "/" +  args.experiment_name + \
                                                         str(current_num) + "_" + str(current_inner_num) \
                                                         + "_best_model.pth"
                intermediate_config["hyper_csv"] = path3 + "/" + args.experiment_name + \
                                                         str(current_num) + "_" + str(current_inner_num) \
                                                         + "hyperparameter.csv"

                intermediate_config["log_path"] = path3 + config["log_path"]

                intermediate_config["singleton_percentage"] = current_num / 100.0
                intermediate_config["singleton_percentage_end"] = current_inner_num / 100.0
                print(intermediate_config["singleton_percentage"])
                print(intermediate_config["singleton_percentage"])
                print(current_num / 100.0)
                print(current_inner_num / 100.0)

                with open(path3 + "/config.json", "w") as config_out:
                    json.dump(intermediate_config, config_out)


