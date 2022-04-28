import numpy as np


def precision_at_k_single_q(queue, k):
    return np.sum(queue[:k]) / k


def recall_at_k_single_q(queue, k):
    denominator = np.sum(queue)
    return np.sum(queue[:k]) / denominator

def precision_at_r_single_q(queue):
    r = np.sum(queue)
    return np.sum(queue[:r]) / r


# expects embeddings(dim num_samples x embedding_size) and labels(dim num_samples), and k_list as a list
# returns 2 lists containing the precision and recall values for the given k's
def calculate_data_set_rec_prec(embeddings, labels, k_list, precision=True, recall=True):
    prec_list = np.zeros(len(k_list))
    recall_list = np.zeros(len(k_list))
    num_samples = 0
    # iterate over all samples
    for curr_index, (embedding, label) in enumerate(zip(embeddings, labels)):
        # get current class indices inf form of a binary array
        true_label_array = (labels - label) == 0
        # see if there is more than one sample of the class, proceed if this is the case
        if np.sum(true_label_array) > 1:
            num_samples += 1
            # get distances between embeddings
            distance = np.linalg.norm(embeddings - embedding, axis=1)
            # put current samples at the end of the queue so it is not counted by setting distance to inf
            distance[curr_index] = np.inf
            # sort the queue
            sorted_indices = np.argsort(distance)
            sorted_true_label_array = true_label_array[sorted_indices]
            # compute vaöues for the sample according to the given k values
            for k in range(len(k_list)):
                if precision:
                    prec_list[k] += precision_at_k_single_q(sorted_true_label_array[:-1], k_list[k])
                if recall:
                    recall_list[k] += recall_at_k_single_q(sorted_true_label_array[:-1], k_list[k])
        if curr_index % 1000==0:
            print("Processed " + curr_index + " samples")
    return prec_list / num_samples, recall_list / num_samples


if __name__ == "__main__":
    #rand_emb = np.random.rand(20, 128)
    #rand_labels = np.random.random_integers(0, 6, (20))
    k_list = [1, 2, 5, 10, 20, 25, 30, 40, 50]
    #print(rand_labels.shape)
    #print(rand_emb.shape)
    embeddings = np.load("C:/Users/nicmu/Desktop/sebastian_thesis_tests/seb_thesis_512_embeddings_test.npy")
    labels = np.load("C:/Users/nicmu/Desktop/sebastian_thesis_tests/seb_thesis_512_labels_test.npy")

    print(calculate_data_set_rec_prec(embeddings, labels, k_list, precision=True, recall=True))
