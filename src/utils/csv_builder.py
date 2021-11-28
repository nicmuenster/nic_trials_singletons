import os, sys
import pandas as pd
import numpy as np
import argparse

def remove_other_entries(csv_file_path, new_path=None):
    data_frame = pd.read_csv(csv_file_path)
    label_list = data_frame['Label'].to_list()
    name_list = data_frame['Image Name'].to_list()
    new_pd = pd.DataFrame({'Image Name': name_list, 'Label': label_list})
    if new_path:
        new_pd.to_csv(new_path)
    else:
        new_pd.to_csv(csv_file_path)


def split_singulars_chest(orig_path, dest_many, dest_singles, rename_dict=None):
    data_frame = pd.read_csv(orig_path)
    if rename_dict:
        data_frame.rename({rename_dict['Image Name']: 'Image Name', rename_dict['Label']: 'Label'}, axis=1, inplace=True)
    col_id_arr = data_frame['Label'].to_numpy()
    ids, id_counts = np.unique(col_id_arr, return_counts=True)
    single_image_ids = ids[id_counts == 1]
    multiple_image_ids = ids[id_counts > 1]
    single_files = data_frame.loc[data_frame['Label'].isin(single_image_ids)]
    many_files = data_frame.loc[data_frame['Label'].isin(multiple_image_ids)]
    single_files.to_csv(dest_singles,  header=True)
    many_files.to_csv(dest_many, header=True)

def testset_split(file_path, dest_path, singles=False):
    data_frame = pd.read_csv(file_path)
    col_id_arr = data_frame['Label'].to_numpy()
    ids, id_counts = np.unique(col_id_arr, return_counts=True)
    ordered_ids = ids[id_counts.argsort()]
    reversed_ordered_ids = ordered_ids[::-1]
    train_ids = []
    test_ids = []
    split_counter = 0
    for id in reversed_ordered_ids:
        if split_counter == 1:
            test_ids.append(id)
            split_counter = 0
        else:
            train_ids.append(id)
            split_counter = split_counter + 1

    train_files = data_frame.loc[data_frame['Label'].isin(train_ids)].sort_values('Image Name')
    test_files = data_frame.loc[data_frame['Label'].isin(test_ids)].sort_values('Image Name')
    if singles:
        train_files.to_csv(dest_path + 'test_singles.csv', header=True)
        test_files.to_csv(dest_path + 'training_data_singles.csv', header=True)
    else:
        train_files.to_csv(dest_path + 'test.csv', header=True)
        test_files.to_csv(dest_path + 'training_data.csv', header=True)

def get_statistics(csv_path):
    data_frame = pd.read_csv(csv_path)
    col_id_arr = data_frame['Label'].to_numpy()
    ids, id_counts = np.unique(col_id_arr, return_counts=True)
    print({"Num Images": len(col_id_arr), "Num Classes": len(ids)})


def fold_split(file_path, dest_path, num_folds, singles=False):
    data_frame = pd.read_csv(file_path)
    col_id_arr = data_frame['Label'].to_numpy()
    ids, id_counts = np.unique(col_id_arr, return_counts=True)
    ordered_ids = ids[id_counts.argsort()]
    reversed_ordered_ids = ordered_ids[::-1]
    fold_list = [[] for x in range(num_folds)]
    split_counter = 0
    for id in reversed_ordered_ids:
        if split_counter == num_folds:
            split_counter = 0
        fold_list[split_counter].append(id)
        split_counter = split_counter + 1
    for fold in range(num_folds):
        train = []
        val = fold_list[fold]
        for x in range(num_folds):
            if x == fold:
                pass
            else:
                train = train + fold_list[x]
        train_files = data_frame.loc[data_frame['Label'].isin(train)].sort_values(
            'Image Name')
        val_files = data_frame.loc[data_frame['Label'].isin(val)].sort_values('Image Name')
        if singles:
            train_files.to_csv(dest_path + 'train_singles' + str(fold+1) + '.csv', header=True)
            val_files.to_csv(dest_path + 'val_singles' + str(fold+1) + '.csv', header=True)
        else:
            train_files.to_csv(dest_path + 'train' + str(fold+1) + '.csv', header=True)
            val_files.to_csv(dest_path + 'val' + str(fold+1) + '.csv', header=True)


if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('csv_builder')
    parser.add_argument('--folder_path', default='./',
                        help='the path to where the folder should be created')
    parser.add_argument('--csv_path', default='./Data_Entry_2017_v2020.csv',
                        help='the path to the standard csv needed for buiding')
    parser.add_argument("-f", "--num_folds", type=int, default=5,
                        help="The number of subfolds to create")
    parser.add_argument('--experiment_name', default='chestxray',
                        help='name of the subfolder structure')


    args = parser.parse_args()
    rename_dict = None
    if args.experiment_name == 'chestxray':
        rename_dict = {'Image Name': 'Image Index', 'Label': 'Patient ID'}
    folder_path = args.folder_path + args.experiment_name
    if not (os.path.exists(folder_path + '_csv')):
        os.mkdir(folder_path + '_csv')

    orig_path = args.csv_path
    dest_many = "./many_images.csv"
    dest_singles = "./single_images.csv"
    split_singulars_chest(orig_path, dest_many, dest_singles, rename_dict=rename_dict)
    testset_split(dest_singles, dest_path='./', singles=True)
    testset_split(dest_many, dest_path='./', singles=False)
    fold_split(file_path='./training_data.csv', dest_path='./chestxray_csv/', num_folds=5, singles=False)
    fold_split(file_path='./training_data_singles.csv', dest_path='./chestxray_csv/', num_folds=5, singles=True)
    for file in os.listdir('./chestxray_csv/'):
        remove_other_entries('./chestxray_csv/' + file, new_path=None)
        print(file)
        get_statistics('./chestxray_csv/' + file)
    print("test")
    get_statistics('./test.csv')
    print("test_singles")
    get_statistics('./test_singles.csv')




