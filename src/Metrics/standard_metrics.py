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
# returns
''' Approach that used lists, not quite necessary for this case, but maybe also useful at some point
def calculate_data_set_rec_prec(embeddings, labels, k_list, precision=True, recall=True):
    prec_list = []
    recall_list = []
    for _ in range(len(k_list)):
        prec_list.append([])
        recall_list.append([])
        
    for embedding, label in zip(embeddings, labels):
        distance = np.linalg.norm(embeddings - embedding)
        true_label_array = (labels - label) == 0
        sorted_indices = np.argsort(distance)
        sorted_true_label_array = true_label_array[sorted_indices]
        for k in range(len(k_list)):
            if precision:
                prec_list[k].append(precision_at_k_single_q(sorted_true_label_array, k_list[k]))
            if recall:
                recall_list[k].append(recall_at_k_single_q(sorted_true_label_array, k_list[k]))
'''

def calculate_data_set_rec_prec(embeddings, labels, k_list, precision=True, recall=True):
    prec_list = np.zeros(len(k_list))
    recall_list = np.zeros(len(k_list))
    num_samples = 0
    for curr_index, (embedding, label) in enumerate(zip(embeddings, labels)):
        true_label_array = (labels - label) == 0
        if np.sum(true_label_array) > 1:
            num_samples += 1
            distance = np.linalg.norm(embeddings - embedding, axis=1)
            distance[curr_index] = np.inf
            sorted_indices = np.argsort(distance)
            sorted_true_label_array = true_label_array[sorted_indices]
            for k in range(len(k_list)):
                if precision:
                    prec_list[k] += precision_at_k_single_q(sorted_true_label_array[:-1], k_list[k])
                if recall:
                    recall_list[k] += recall_at_k_single_q(sorted_true_label_array[:-1], k_list[k])

    return prec_list / num_samples, recall_list / num_samples


if __name__ == "__main__":
    rand_emb = np.random.rand(20, 128)
    rand_labels = np.random.random_integers(0, 6, (20))
    rand_k_list = [2, 5, 10]
    print(rand_labels.shape)
    print(rand_emb.shape)

    print(calculate_data_set_rec_prec(rand_emb, rand_labels, rand_k_list, precision=True, recall=True))
