import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# light weight batch miner without debugging tools
def sample_func(current_id, num_samples, name_array, id_array):
    candidates = name_array[current_id == id_array]
    # print(candidates)
    num_samples = int(num_samples)
    # print(num_samples)
    samples = np.random.choice(candidates, num_samples, replace=False)
    filter_mask = np.isin(name_array, samples, invert=True)
    id_array = id_array[filter_mask]
    name_array = name_array[filter_mask]
    return samples, name_array, id_array


def sample_singles(single_name_array, single_id_array, num_samples):
    sample_idxs = np.random.choice(range(len(single_name_array)), num_samples, replace=False)
    sampled_names = single_name_array[sample_idxs]
    sampled_ids = single_id_array[sample_idxs]
    filter_mask_names = np.isin(single_name_array, sampled_names, invert=True)
    filter_mask_ids = np.isin(single_id_array, sampled_ids, invert=True)
    single_id_array = single_id_array[filter_mask_ids]
    single_name_array = single_name_array[filter_mask_names]
    return sampled_names, sampled_ids, single_name_array, single_id_array


def get_block_with_singles(bucket_size, name_array, id_array, single_name_array, single_id_array):
    operation_successful = True
    block = []
    current_id = np.random.choice(id_array)
    ids, counts = np.unique(id_array, return_counts=True)
    current_count = counts[np.argwhere(ids == current_id)]
    # case of exactly 8 samples
    if current_count % bucket_size == 0:
        samples, name_array, id_array = sample_func(current_id, bucket_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))
        return operation_successful, block, name_array, id_array, single_name_array, single_id_array
    # more than 8 images with this id
    if current_count > bucket_size:
        # check whether its 9 or mod8 == 1
        if current_count % 8 == 1:
            samples, name_array, id_array = sample_func(current_id, 2, name_array, id_array)
            for sample in samples:
                block.append((sample, current_id))
        # 50/50 whether we fill the block with 8 or num%8
        else:
            if random.random() < 0.5:
                # fill with 8
                samples, name_array, id_array = sample_func(current_id, bucket_size, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
                return operation_successful, block, name_array, id_array, single_name_array, single_id_array
            else:
                # case 7, or more generally if there is not enough space for another pair we only take bucket_size - 3, or use 7 and fill it up with a single
                if current_count % bucket_size == bucket_size - 1:
                    # if there are still singles, we fill it up
                    if single_id_array.size != 0:
                        # first the 7
                        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array,
                                                                    id_array)
                        for sample in samples:
                            block.append((sample, current_id))
                        # then one single
                        sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(
                            single_name_array,
                            single_id_array, 1)

                        for name, id in zip(sampled_names, sampled_ids):
                            block.append((name, id))
                        return operation_successful, block, name_array, id_array, single_name_array, single_id_array
                    else:
                        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size - 2,
                                                                    name_array, id_array)
                        for sample in samples:
                            block.append((sample, current_id))
                # all other cases
                else:
                    samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array,
                                                                id_array)
                    for sample in samples:
                        block.append((sample, current_id))
    else:
        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))
    rest = bucket_size - len(block)

    while rest != 0:
        ids, counts = np.unique(id_array, return_counts=True)
        shortened_counts = np.mod(counts, bucket_size)
        # filter possible ids, so   1 < num_ids and rest - num_ids < 2
        possible_ids = ids[
            (shortened_counts > 1) & (((rest - shortened_counts) > 1) | ((rest - shortened_counts) == 0))]
        # no combinations available
        if possible_ids.size == 0:
            possible_ids = ids[shortened_counts == bucket_size - 1]
            # not even when splitting up 7s building Block failed
            if possible_ids.size == 0:
                # fill up with singles
                if single_id_array.size >= rest:

                    sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                    single_id_array,
                                                                                                    rest)
                    for name, id in zip(sampled_names, sampled_ids):
                        block.append((name, id))

                    return operation_successful, block, name_array, id_array, single_name_array, single_id_array

                else:
                    operation_successful = False
            # fill up with parts of a seven --> operation successful
            else:
                current_id = np.random.choice(possible_ids)
                samples, name_array, id_array = sample_func(current_id, rest, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
            return operation_successful, block, name_array, id_array, single_name_array, single_id_array
        # otherwise continue filling til rest == 0
        else:
            current_id = np.random.choice(possible_ids)
            samples, name_array, id_array = sample_func(current_id, shortened_counts[ids == current_id], name_array,
                                                        id_array)
            for sample in samples:
                block.append((sample, current_id))
            rest = bucket_size - len(block)
        # print(rest)
        # the loop ended, so we reached blocksize 0 and were successful
    return operation_successful, block, name_array, id_array, single_name_array, single_id_array


def build_mining_list_32(csv_file_path, block_size, csv_singles, singleton_percentage=1.0,
                         singleton_percentage_end=0.0, keep_singletons=True):
    patient_data_frame = pd.read_csv(csv_file_path)
    name_array = patient_data_frame['Image Name'].to_numpy()
    id_array = patient_data_frame['Label'].to_numpy()
    mining_list = []
    next_eight = []
    garbage_collector = []
    single_rest = []
    many_rest = []
    reserved_singleton_names = []
    reserved_singleton_ids = []
    list_counter = 0
    row_order = [0, 4, 2, 6, 1, 5, 3, 7]

    # extract singles
    patient_data_frame = pd.read_csv(csv_singles)
    single_name_array = patient_data_frame['Image Name'].to_numpy()
    single_id_array = patient_data_frame['Label'].to_numpy()
    cutoff = int(len(single_name_array) * singleton_percentage)
    single_name_array = single_name_array[:cutoff]
    single_id_array = single_id_array[:cutoff]
    if singleton_percentage_end > 0.0:
        sec_cutoff = int(len(single_name_array) * singleton_percentage_end)
        reserved_singleton_names = single_name_array[:sec_cutoff]
        reserved_singleton_ids = single_id_array[:sec_cutoff]
        single_name_array = single_name_array[sec_cutoff:]
        single_id_array = single_id_array[sec_cutoff:]

    num_many = len(name_array)
    num_singles = len(single_name_array)
    partial_probability = num_singles / (num_many + num_singles)
    fitted_probability = partial_probability * 4 * 0.9  # because it is the probabiliy of one block of four and other sampling
    changing_prob = fitted_probability
    while name_array.size != 0:
        # first block -> chance to sample a singleton block
        if list_counter == 0 and single_id_array.size >= block_size:
            if random.random() < changing_prob:
                # random chance to insert a block of blocksize singles
                changing_prob -= 1
                single_block = []
                sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                single_id_array,
                                                                                                block_size)
                for name, id in zip(sampled_names, sampled_ids):
                    single_block.append((name, id))
                next_eight.append(single_block)
                list_counter += 1

                while random.random() < changing_prob and single_id_array.size >= block_size:
                    # random chance to insert a block of blocksize singles
                    changing_prob -= 1
                    single_block = []
                    sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                    single_id_array,
                                                                                                    block_size)
                    for name, id in zip(sampled_names, sampled_ids):
                        single_block.append((name, id))
                    next_eight.append(single_block)
                    list_counter += 1
            else:
                operation_successful, block, name_array, id_array, single_name_array, single_id_array = get_block_with_singles(
                    block_size, name_array, id_array,
                    single_name_array, single_id_array)
                # print(changing_prob)
                if not operation_successful:
                    for part in block:
                        garbage_collector.append(part)
                else:
                    next_eight.append(block)
                    list_counter += 1

        else:
            operation_successful, block, name_array, id_array, single_name_array, single_id_array = get_block_with_singles(
                block_size, name_array, id_array,
                single_name_array, single_id_array)
            # print(changing_prob)
            if not operation_successful:
                for part in block:
                    garbage_collector.append(part)
            else:
                next_eight.append(block)
                list_counter += 1

            # once 32 is reached, the samples get shuffled and appended
        if len(next_eight) == 4:
            single_rest.append(single_name_array.size)
            many_rest.append(name_array.size)
            for x in row_order:
                for y in range(len(next_eight)):
                    # print(next_eight[y][x])
                    mining_list.append(next_eight[y][x])
            list_counter = 0
            changing_prob = fitted_probability

            next_eight = []

    if next_eight:
        for eight in next_eight:
            for one in eight:
                mining_list.append(one)

    if garbage_collector:
        for garbage in garbage_collector:
            mining_list.append(garbage)

    if keep_singletons:
        # append the reserved singles to the end
        single_id_array = np.array(list(single_id_array) + list(reserved_singleton_ids))
        single_name_array = np.array(list(single_name_array) + list(reserved_singleton_names))

    if single_id_array.size != 0:
        sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                        single_id_array,
                                                                                        single_id_array.size)
        for name, id in zip(sampled_names, sampled_ids):
            mining_list.append((name, id))

    print("successfully created mining list")
    return mining_list


def get_block_with_singles_debug(bucket_size, name_array, id_array, single_name_array, single_id_array,
                                 sampled_singles_count, check_remain):
    if single_name_array.size <= bucket_size:
        if check_remain:
            print("No more Singles, remaining others:", name_array.size)
            check_remain = False
    operation_successful = True
    block = []
    current_id = np.random.choice(id_array)
    ids, counts = np.unique(id_array, return_counts=True)
    current_count = counts[np.argwhere(ids == current_id)]
    # case of exactly 8 samples
    if current_count % bucket_size == 0:
        samples, name_array, id_array = sample_func(current_id, bucket_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))
        return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count
    # more than 8 images with this id
    if current_count > bucket_size:
        # check whether its 9 or mod8 == 1
        if current_count % 8 == 1:
            samples, name_array, id_array = sample_func(current_id, 2, name_array, id_array)
            for sample in samples:
                block.append((sample, current_id))
        # 50/50 whether we fill the block with 8 or num%8
        else:
            if random.random() < 0.5:
                # fill with 8
                samples, name_array, id_array = sample_func(current_id, bucket_size, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
                return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count
            else:
                # case 7, or more generally if there is not enough space for another pair we only take bucket_size - 3, or use 7 and fill it up with a single
                if current_count % bucket_size == bucket_size - 1:
                    # if there are still singles, we fill it up
                    if single_id_array.size != 0:
                        # first the 7
                        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array,
                                                                    id_array)
                        for sample in samples:
                            block.append((sample, current_id))
                        # then one single
                        sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(
                            single_name_array,
                            single_id_array, 1)
                        sampled_singles_count += 1
                        for name, id in zip(sampled_names, sampled_ids):
                            block.append((name, id))
                        return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count
                    else:
                        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size - 2,
                                                                    name_array, id_array)
                        for sample in samples:
                            block.append((sample, current_id))
                # all other cases
                else:
                    samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array,
                                                                id_array)
                    for sample in samples:
                        block.append((sample, current_id))
    else:
        samples, name_array, id_array = sample_func(current_id, current_count % bucket_size, name_array, id_array)
        for sample in samples:
            block.append((sample, current_id))
    rest = bucket_size - len(block)

    while rest != 0:
        ids, counts = np.unique(id_array, return_counts=True)
        shortened_counts = np.mod(counts, bucket_size)
        # filter possible ids, so   1 < num_ids and rest - num_ids < 2
        possible_ids = ids[
            (shortened_counts > 1) & (((rest - shortened_counts) > 1) | ((rest - shortened_counts) == 0))]
        # no combinations available
        if possible_ids.size == 0:
            possible_ids = ids[shortened_counts == bucket_size - 1]
            # not even when splitting up 7s building Block failed
            if possible_ids.size == 0:
                # fill up with singles
                if single_id_array.size >= rest:

                    sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                    single_id_array,
                                                                                                    rest)
                    for name, id in zip(sampled_names, sampled_ids):
                        block.append((name, id))
                    sampled_singles_count += rest

                    return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count

                else:
                    operation_successful = False
            # fill up with parts of a seven --> operation successful
            else:
                current_id = np.random.choice(possible_ids)
                samples, name_array, id_array = sample_func(current_id, rest, name_array, id_array)
                for sample in samples:
                    block.append((sample, current_id))
            return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count
        # otherwise continue filling til rest == 0
        else:
            current_id = np.random.choice(possible_ids)
            samples, name_array, id_array = sample_func(current_id, shortened_counts[ids == current_id], name_array,
                                                        id_array)
            for sample in samples:
                block.append((sample, current_id))
            rest = bucket_size - len(block)
        # print(rest)
        # the loop ended, so we reached blocksize 0 and were successful
    return operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count


def build_mining_list_32_debug(csv_file_path, block_size, csv_singles, check_remain=True, singleton_percentage=1.0,
                               singleton_percentage_end=0.0, keep_singletons=True, savepath=None):
    patient_data_frame = pd.read_csv(csv_file_path)
    name_array = patient_data_frame['Image Name'].to_numpy()
    id_array = patient_data_frame['Label'].to_numpy()
    mining_list = []
    next_eight = []
    garbage_collector = []
    single_rest = []
    many_rest = []
    reserved_singleton_names = []
    reserved_singleton_ids = []
    list_counter = 0
    row_order = [0, 4, 2, 6, 1, 5, 3, 7]

    # extract singles
    sampled_singles_count = 0
    patient_data_frame = pd.read_csv(csv_singles)
    single_name_array = patient_data_frame['Image Name'].to_numpy()
    single_id_array = patient_data_frame['Label'].to_numpy()
    cutoff = int(len(single_name_array) * singleton_percentage)
    single_name_array = single_name_array[:cutoff]
    single_id_array = single_id_array[:cutoff]
    if singleton_percentage_end > 0.0:
        sec_cutoff = int(len(single_name_array) * singleton_percentage_end)
        reserved_singleton_names = single_name_array[:sec_cutoff]
        reserved_singleton_ids = single_id_array[:sec_cutoff]
        single_name_array = single_name_array[sec_cutoff:]
        single_id_array = single_id_array[sec_cutoff:]

    print("num_samples: ", len(name_array) + len(single_name_array))
    print("num_singles: ", len(single_name_array))
    num_many = len(name_array)
    num_singles = len(single_name_array)
    partial_probability = num_singles / (num_many + num_singles)
    fitted_probability = partial_probability * 4 * 0.9  # because it is the probability of one block of four and other sampling
    changing_prob = fitted_probability
    print(changing_prob)
    while name_array.size != 0:
        # first block -> chance to sample a singleton block
        if list_counter == 0 and single_id_array.size >= block_size:
            if random.random() < changing_prob:
                # random chance to insert a block of blocksize singles
                changing_prob -= 1
                single_block = []
                sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                single_id_array,
                                                                                                block_size)
                for name, id in zip(sampled_names, sampled_ids):
                    single_block.append((name, id))
                next_eight.append(single_block)
                list_counter += 1

                while random.random() < changing_prob and single_id_array.size >= block_size:
                    # random chance to insert a block of blocksize singles
                    changing_prob -= 1
                    single_block = []
                    sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                                    single_id_array,
                                                                                                    block_size)
                    for name, id in zip(sampled_names, sampled_ids):
                        single_block.append((name, id))
                    next_eight.append(single_block)
                    list_counter += 1
            else:
                operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count = get_block_with_singles_debug(
                    block_size, name_array, id_array,
                    single_name_array, single_id_array, sampled_singles_count, check_remain)
                # print(changing_prob)
                if not operation_successful:
                    for part in block:
                        garbage_collector.append(part)
                else:
                    next_eight.append(block)
                    list_counter += 1

        else:
            operation_successful, block, name_array, id_array, single_name_array, single_id_array, check_remain, sampled_singles_count = get_block_with_singles_debug(
                block_size, name_array, id_array,
                single_name_array, single_id_array, sampled_singles_count, check_remain)
            # print(changing_prob)
            if not operation_successful:
                for part in block:
                    garbage_collector.append(part)
            else:
                next_eight.append(block)
                list_counter += 1

            # once 32 is reached, the samples get shuffled and appended
        if len(next_eight) == 4:
            single_rest.append(single_name_array.size)
            many_rest.append(name_array.size)
            for x in row_order:
                for y in range(len(next_eight)):
                    # print(next_eight[y][x])
                    mining_list.append(next_eight[y][x])
            list_counter = 0
            changing_prob = fitted_probability

            next_eight = []

    if next_eight:
        for eight in next_eight:
            for one in eight:
                mining_list.append(one)
    remain_garb = len(garbage_collector)
    if garbage_collector:
        print("Garbage_size", len(garbage_collector))
        for garbage in garbage_collector:
            mining_list.append(garbage)
    remain = single_id_array.size

    if keep_singletons:
        # append the reserved singles to the end
        single_id_array = np.array(list(single_id_array) + list(reserved_singleton_ids))
        single_name_array = np.array(list(single_name_array) + list(reserved_singleton_names))

    if single_id_array.size != 0:
        print("Remaining Singles:", single_id_array.size)
        print("Remaining Many:", id_array.size)

        sampled_names, sampled_ids, single_name_array, single_id_array = sample_singles(single_name_array,
                                                                                        single_id_array,
                                                                                        single_id_array.size)
        for name, id in zip(sampled_names, sampled_ids):
            mining_list.append((name, id))
    steps = np.arange(len(single_rest))
    plt.plot(steps, single_rest, 'b-', label="num_singles", color="red")
    plt.plot(steps, many_rest, 'b-', label="num_many", color="blue")
    plt.ylabel("num_images")

    plt.xlabel('batch')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    lgd = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     fancybox=True, shadow=True, ncol=3)
    if savepath is not None:
        plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')

    print("sampled_singles: ", sampled_singles_count)
    plt.show()
    print("successfully created mining list")
    return remain, remain_garb, mining_list

