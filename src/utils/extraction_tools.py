import glob, os
import argparse
import pandas as pd

def get_csv_summary(dir_path, result_folder, key_list):
    result_keys = ["name"] + key_list
    result_df = pd.DataFrame(columns=result_keys)
    avg_df = pd.DataFrame(columns=result_keys)

    for file in glob.glob(dir_path + "**/*.csv", recursive=True):
        print(file)
        iter_frame = pd.read_csv(file)
        name = file.split("/")[-1]
        name = name.split("hyper")[0]
        reduced_iter_frame = iter_frame[key_list].copy()
        # get top 3, get their mean and std, add them to avg_df
        top_results = reduced_iter_frame.sort_values('result').head(3)
        top_std = top_results.std()
        top_mean = top_results.mean()
        top_std['name'] = name + "mean"
        top_mean['name'] = name + "avg"
        avg_list = [avg_df, top_mean, top_std]
        avg_df = pd.concat(avg_list)
        # get best params, add them to result_df
        winner = top_results.iloc[[0]]
        winner['name'] = name
        frame_list = [result_df, winner]
        result_df = pd.concat(frame_list)
    # get mean and std from result_df, add to the end
    res_std = result_df.std()
    res_mean = result_df.mean()
    res_std['name'] = "overall_std"
    res_mean['name'] = "overall_mean"
    final_frame_list = [result_df, res_mean, res_std, avg_df]
    final_result_df = pd.concat(final_frame_list)
    final_result_df.to_csv(result_folder)


if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('inventory_builder')
    parser.add_argument('--dir_path', default='/home/woody/iwi5/iwi5014h/', help='the basic config needed for buiding')
    parser.add_argument('--experiment_name', default='chestxray',
                        help='name of the subfolder structure')
    parser.add_argument('--key_list', nargs='+', default=["learning_rate", "weight_decay",
                                         "neg_margin", "pos_margin", "result"],
                        help='list of keys to extract')

    parser.add_argument('--result_folder', default='~/results/',
                        help='path where results should be saved to')

    args = parser.parse_args()

    if not (os.path.exists(args['result_folder'])):
        os.mkdir(args['result_folder'])
    dir_pth = args["dir_path"] + args["experiment_name"]
    get_csv_summary(dir_pth, args['result_folder'], args["key_list"])
