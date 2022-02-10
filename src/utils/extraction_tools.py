import glob, os
import argparse
import pandas as pd

def get_csv_summary(dir_path, result_folder, key_list, experiment_name):
    result_keys = ["name"] + key_list
    result_df = pd.DataFrame(columns=result_keys)
    avg_df = pd.DataFrame(columns=result_keys)

    for file in glob.glob(dir_path + "/**/*.csv", recursive=True):
        print(file)
        iter_frame = pd.read_csv(file)
        name = file.split("/")[-1]
        name = name.split("hyper")[0]
        print(list(iter_frame.columns))
        reduced_iter_frame = iter_frame[key_list].copy()
        # get top 3, get their mean and std, add them to avg_df
        top_results = reduced_iter_frame.sort_values('result', ascending=False).head(3)
        top_std = top_results.std()
        top_mean = top_results.mean()
        filler_top_std = dict()
        for key in top_std.keys():
            filler_top_std[key] = [top_std[key]]
        top_std = filler_top_std
        #print([key for key in top_mean.keys()])
        filler_top_mean = dict()
        for key in top_mean.keys():
            filler_top_mean[key] = [top_mean[key]]
        top_mean = filler_top_mean

        top_std['name'] = [name + "mean"]
        top_mean['name'] = [name + "avg"]
        top_std = pd.DataFrame(top_std)
        top_mean = pd.DataFrame(top_mean)
        avg_list = [avg_df, top_mean, top_std]
        avg_df = pd.concat(avg_list)
        # get best params, add them to result_df
        winner = top_results.iloc[[0]]
        winner['name'] = name
        frame_list = [result_df, winner]
        result_df = pd.concat(frame_list)
    # get mean and std from result_df, add to the end
    res_std = result_df.std()
    #
    filler_res_std = dict()
    for key in res_std.keys():
        filler_res_std[key] = [res_std[key]]
    res_std = filler_res_std
    #
    res_mean = result_df.mean()
    #
    filler_res_mean = dict()
    for key in res_mean.keys():
        filler_res_mean[key] = [res_mean[key]]
    res_mean = filler_res_mean
    #
    res_std['name'] = ["overall_std"]
    res_mean['name'] = ["overall_mean"]
    res_std = pd.DataFrame(res_std)
    res_mean = pd.DataFrame(res_mean)
    final_frame_list = [result_df, res_mean, res_std, avg_df]
    final_result_df = pd.concat(final_frame_list)
    final_result_df.to_csv(result_folder + experiment_name + ".csv")


if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('inventory_builder')
    parser.add_argument('--dir_path', default='/home/woody/iwi5/iwi5014h/', help='the basic config needed for buiding')
    parser.add_argument('--experiment_name', default='chestxray',
                        help='name of the subfolder structure')
    parser.add_argument('--key_list', nargs='+', default=["learning_rate", "weight_decay",
                                         "neg_margin", "pos_margin", "result"],
                        help='list of keys to extract')

    parser.add_argument('--result_folder', default='./results/',
                        help='path where results should be saved to')

    args = parser.parse_args()

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)
    dir_pth = args.dir_path + args.experiment_name + "/trials"
    get_csv_summary(dir_pth, args.result_folder, args.key_list, args.experiment_name)
