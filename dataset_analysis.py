import h5py
import numpy as np
import random
import time
from datetime import date
# import matplotlib.pyplot as plt
import json
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, balanced_accuracy_score, \
    confusion_matrix, f1_score


def analyze_distribution(print_info, plot_info):
    """Analyzes the label distribution for the given dataset."""
    total_res = {}
    total_sum = 0
    with open('../data/binding_sites_seq.fasta', 'r') as fasta:
        lines_f = fasta.readlines()
        id = ""
        for line in lines_f:
            if line.startswith(">"):
                id = line.split(">")[1].strip()
            else:
                total_res[id] = len(line)
                total_sum += len(line)

    binding_res = {}
    binding_sum = 0
    with open('../data/binding_residues.txt', 'r') as binding:
        lines_b = binding.readlines()
        for line in lines_b:
            id = line.split("\t")[0]
            binding_res[id] = len(line.split(","))
            binding_sum += len(line.split(","))

    ratios = {}
    for key in total_res.keys():
        ratios[key] = (float(binding_res[key]) / float(total_res[key]))
    overall_ratio = float(binding_sum) / float(total_sum)

    if print_info:
        print("Ratios:")
        for key in ratios.keys():
            print(str(ratios[key]))
        print("\nAll: " + str(total_sum))
        print("Binding: " + str(binding_sum))
        print("= " + str(overall_ratio))

    # plotting
    """
    if plot_info:
        ratio_values = sorted(ratios.values())
        mod_ratios = []
        max_length = max(total_res.values())
        max_ratio = max(ratios.values())
        for key in total_res.keys():
            mod_ratios.append(ratios[key] * (total_res[key] / total_sum) * len(ratio_values))
        mod_ratios = sorted(mod_ratios)
        mean1 = sum(ratio_values) / len(ratio_values)
        if print_info:
            print("Mean 1: " + str(mean1))
            print("Sum of ratios 1: " + str(sum(ratio_values)))
        mean2 = sum(mod_ratios) / len(mod_ratios)
        if print_info:
            print("Mean 2: " + str(mean2))
            print("Sum of ratios 2: " + str(sum(mod_ratios)))
        x = range(0, len(ratio_values))
        plt.scatter(x=x, y=ratio_values, s=0.4)
        plt.scatter(x=x, y=mod_ratios, s=0.4, c='green')
        plt.axhline(mean1, c='red')
        plt.axhline(mean2, c='yellow')
        plt.show()
        """

    return total_sum, overall_ratio, total_res, ratios


def find_one_split(embeddings, pos_to_neg, sum_of_res, ratios, diff, split):
    """Finds an approximately fitting split for a given number of residues and a given class ratio."""
    final_ratio = 0
    limit = (sum_of_res * split)
    # repeat as long as ratio deviates too much
    while final_ratio >= (pos_to_neg + diff) or final_ratio < (pos_to_neg - diff):
        ratio_so_far = 0
        sum_of_ratios = 0
        residues_so_far = 0
        dict_list = list(embeddings.items())
        collected_ids = []
        # collect proteins and benefit ratio
        while residues_so_far < limit:
            id, matrix = random.choice(dict_list)
            if id not in collected_ids:
                if (ratio_so_far > pos_to_neg > ratios[id]) or (ratio_so_far < pos_to_neg < ratios[id]):
                    residues_so_far += len(matrix)
                    collected_ids.append(id)
                    sum_of_ratios += ratios[id]
                    ratio_so_far = float(sum_of_ratios) / float(len(collected_ids))

        # identify final ratio
        final_ratio = ratio_so_far

    # remove found proteins from embeddings (IMPORTANT!!!)
    for id in collected_ids:
        del embeddings[id]

    print("Class ratio : " + str(final_ratio) + "\t Dataset proportion : " + str(residues_so_far / sum_of_res))
    print("\n------ TOTAL ONE SPLIT ------\n")
    print("Residues : " + str(residues_so_far) + "\tProteins : " + str(len(collected_ids)))
    print("\n------------\n")
    return collected_ids, residues_so_far, embeddings


def find_splits_parallel(embeddings, pos_to_neg, sum_of_res, total_res, ratios, diff, k):
    """Finds approximately fitting splits in a rather parallel manner instead of sequentially."""
    n = len(embeddings)
    final_ratios = [0.0] * k
    limit = (sum_of_res * (1 / k))
    all_residues_so_far = [0] * k
    all_collected_ids = [""] * k

    while True:  # repeat as long as ratios deviate too much
        do_continue = False
        for final_ratio in final_ratios:
            if final_ratio >= (pos_to_neg + diff) or final_ratio < (pos_to_neg - diff):
                do_continue = True
                break
        if not do_continue:
            break

        total_collected_res = 0
        total_collected_prot = 0
        all_ratios_so_far = [0.0] * k
        all_sums_of_ratios = [0.0] * k
        all_residues_so_far = [0] * k
        dict_list = list(embeddings.items())
        all_collected_ids = [""] * k
        for i in range(0, k):  # initialize list correctly
            all_collected_ids[i] = []
        all_mod_ratios = [0.0] * k
        for i in range(0, k):  # initialize list correctly
            all_mod_ratios[i] = []

        while True:  # collect proteins as long as necessary
            do_continue = False
            for residues_so_far in all_residues_so_far:
                if residues_so_far < (limit * 0.918):  # use limit to adapt how much data to be used, max_used: 0.915
                    do_continue = True
                    break
            if not do_continue:
                break

            # perform the same procedure k times (= for each split)
            for i in range(0, k):
                while True:  # try until found id that was "not yet seen and ratio is helpful"
                    id, matrix = random.choice(dict_list)

                    do_continue = True
                    for collected_ids in all_collected_ids:
                        if id in collected_ids:
                            do_continue = False
                            break
                    if not do_continue:
                        continue  # try again!

                    ratio_so_far = all_ratios_so_far[i]  # debug
                    ratio_in_question = ratios[id]  # debug
                    if (all_ratios_so_far[i] >= pos_to_neg >= ratios[id]) \
                            or (all_ratios_so_far[i] <= pos_to_neg <= ratios[id]):
                        all_residues_so_far[i] += len(matrix)
                        total_collected_res += len(matrix)
                        total_collected_prot += 1
                        all_collected_ids[i].append(id)

                        mod_ratio = ratios[id] * (total_res[id])
                        all_mod_ratios[i].append(mod_ratio)
                        all_sums_of_ratios[i] += mod_ratio
                        all_ratios_so_far[i] = float(all_sums_of_ratios[i]) / float(all_residues_so_far[i])
                        break
                    else:
                        continue  # try again!
        # save ratios
        for i in range(0, k):
            final_ratios[i] = all_ratios_so_far[i]

    # no removal of found proteins from embeddings?
    for i in range(0, k):
        print(f"\nSplit [{i}]")
        print("Class ratio : " + str(final_ratios[i])
              + "\tDataset Proportion : " + str(all_residues_so_far[i] / sum_of_res))
        print("Residues : " + str(all_residues_so_far[i]) + "\tProteins :  " + str(len(all_collected_ids[i])))
        print(all_collected_ids[i])
    print("\n------ TOTAL TRAINING ------\n")
    print("Residues : " + str(total_collected_res) + "\tProteins : " + str(total_collected_prot))
    return all_collected_ids, total_collected_res, total_collected_prot


def create_splits(write_file):
    """Creates splits for cross-validation manually."""

    # (1) read embeddings

    embeddings = dict()
    with h5py.File('../data/baseline_embeddings_binding_sites.h5', 'r') as file:
        for key in file.keys():
            embeddings[key] = np.array(file[key], dtype=np.float16)

    # (2) get class ratio

    total_sum, pos_to_neg, total_res, ratios = analyze_distribution(False, False)

    # (3) find test split (naive)

    print("Test split:")
    start = time.time()
    collected_ids, residues_found, embeddings = find_one_split(embeddings, pos_to_neg, total_sum, ratios, 0.015, 0.2)
    end = time.time()
    mod_total_sum = total_sum - residues_found
    print(f"In {round(end - start, 5)} seconds: " + str(collected_ids) + "\n")

    # (4) find training/validation splits (naive)

    all_collected_ids, total_collected_res, total_collected_prot \
        = find_splits_parallel(embeddings, pos_to_neg, mod_total_sum, total_res, ratios, 0.02, 5)
    print(
        "OVERALL EFFICIENCY : " + str(float(len(collected_ids) + total_collected_prot) / len(total_res)) + " Proteins,"
        + "\t" + str(float(residues_found + total_collected_res) / total_sum) + " Residues")

    eff = round(float(residues_found + total_collected_res) / total_sum, 4)
    if write_file and eff >= 0.97:  # set minimal requirement for efficiency
        all_collected_ids.insert(0, collected_ids)
        write_split_file(all_collected_ids, eff)


def write_split_file(all_ids, eff):
    """Writes the split information into a JSON file."""
    eff = str(eff).replace(".", "-")

    today = date.today()
    now = time.localtime()
    t = time.strftime("%H-%M", now)
    d = today.strftime("%y-%m-%d")
    json.dump(all_ids, open(f"split_{d}_{t}_{eff}.json", "w"))


def read_labels_from_files(file_name):
    """Converts the information of the binding file into a dictionary."""
    binding_pos = {}
    with open(file_name, 'r') as binding_file:
        lines_b = binding_file.readlines()
        for line in lines_b:
            id = line.split("\t")[0]
            binding_res = line.split("\t")[1].strip().split(",")
            binding_pos[id] = binding_res
    return binding_pos


def get_labels(binding_pos, prot_length):
    """For a given list of binding positions and the length of the respective protein, a labels np array is returned."""
    array = np.zeros(shape=prot_length)
    for p in binding_pos:
        array[int(p) - 1] = 1  # do not forget to transform to 0 based!
    return array


def build_splits(ids_file, embeddings_file, binding_file):
    """For given lists of proteins in JSON format, the embeddings are assigned to different splits."""
    # collect binding residues
    binding_pos = read_labels_from_files(binding_file)

    # load embeddings
    ids = json.load(open(ids_file, 'r'))
    embeddings = dict()
    with h5py.File(embeddings_file, 'r') as file:
        for key in file.keys():
            embeddings[key] = np.array(file[key], dtype=np.float16)

    test_set = np.array([])
    test_set_labels = np.array([])
    splits = []
    splits_labels = []
    is_test_split = True

    for split in ids:  # for each split
        add_to_this = np.zeros(shape=(0, 1024))
        add_to_this_labels = np.zeros(shape=0)
        for prot_id in split:  # search each protein of the splits
            # build embeddings/samples vector
            protein = embeddings[prot_id]
            add_to_this = np.concatenate((add_to_this, protein), axis=0)
            # build binding/labels vector
            labels = get_labels(binding_pos[prot_id], len(protein))
            add_to_this_labels = np.concatenate((add_to_this_labels, labels), axis=0)
        if is_test_split:  # test split
            is_test_split = False
            test_set = add_to_this.copy()
            test_set_labels = add_to_this_labels.copy()
        else:
            splits.append(add_to_this.copy())
            splits_labels.append(add_to_this_labels.copy())

    # print results
    print("Test: ")
    # print(str(test_set) + "\t")
    print(str(len(test_set)))
    # print(str(test_set_labels) + "\t")
    print(str(len(test_set_labels)))
    for i in range(0, len(splits)):
        print(f"Split {i}: ")
        # print(str(splits[i]) + "\t")
        print(str(len(splits[i])))
        # print(str(splits_labels[i]) + "\t")
        print(str(len(splits_labels[i])))

    return test_set, test_set_labels, splits, splits_labels


def calculate_se(splits_json, embeddings_file, y_pred, y_test, split):
    """Calculates the standard error (SE) out of the per-residue distribution, without bootstrapping."""
    # load data
    ids = json.load(open(splits_json, 'r'))
    test_split = ids[split]
    embeddings = dict()
    with h5py.File(embeddings_file, 'r') as file:
        for key in file.keys():
            embeddings[key] = np.array(file[key], dtype=np.float16)

    # collect lengths
    test_lengths = []
    for prot_id in test_split:
        length = len(embeddings[prot_id])
        test_lengths.append(length)

    # calculate performance metrics for each protein
    metrics = calc_metrics_for_prots(test_lengths, y_test, y_pred)

    # calculate the standard errors based on the calculated values for the performance metrics
    # -> formula: se = sd/sqrt(n-1)    where n = number of proteins, and sd = standard deviation
    standard_errors = []
    n = len(test_lengths)
    for metric in metrics:
        standard_errors.append(np.std(metric)/np.sqrt(n-1))
    return standard_errors


def calc_metrics_for_prots(lengths, y_test, y_pred):
    """Calculates the different metrics for each protein, given its length."""
    metrics = [[], [], [], [], [], []]    # mcc, acc, prec, rec, bal_acc, f1
    start = 0
    for length in lengths:
        curr_prot_pred = y_pred[start:(start + length)]
        curr_prot_true = y_test[start:(start + length)]
        cm = confusion_matrix(curr_prot_true, curr_prot_pred)
        metrics[0].append(matthews_corrcoef(y_true=curr_prot_true, y_pred=curr_prot_pred))
        metrics[1].append(accuracy_score(y_true=curr_prot_true, y_pred=curr_prot_pred))
        metrics[2].append(precision_score(y_true=curr_prot_true, y_pred=curr_prot_pred))
        metrics[3].append(recall_score(y_true=curr_prot_true, y_pred=curr_prot_pred))
        metrics[4].append(balanced_accuracy_score(y_true=curr_prot_true, y_pred=curr_prot_pred))
        metrics[5].append(f1_score(y_true=curr_prot_true, y_pred=curr_prot_pred))
        start += length
    return metrics


def random_baseline(n):
    """Performs a prediction assigning labels at random, based on their distribution in the dataset."""
    probabilities = np.array([0.9185, 0.0815])
    y_pred_random = np.array(shape=n)
    for i in range(0, n):
        y_pred_random[i] = np.random.choice(a=[0, 1], p=probabilities)
    return y_pred_random


def test():
    with open("pred-and-truth.txt", "r") as f:
        # recollect pred and truth
        c = 0
        pred = []
        truth = []
        lines = f.readlines()
        is_truth = False
        for line in lines:
            if line.strip() == "":
                # ignore empty lines
                continue
            c += 1
            if not line.startswith("-"):
                if not is_truth:
                    pred.append(float(line.strip()))
                else:
                    truth.append(float(line.strip()))
            else:
                is_truth = True
        y_pred = np.array(pred)
        y_test = np.array(truth)
        standard_errors = calculate_se('../dataset-analysis/split_21-06-04_13-07_0-9668.json',
                                       '../data/MSA1_binding_sites.h5', y_pred, y_test)


if __name__ == '__main__':
    # analyze_distribution(True, True)     # only call for analyzing purposes

    # WORKFLOW (call separately)

    # (1) perform splitting the dataset
    create_splits(write_file=True)

    # (2) convert found split into input
    # X_test, y_test, all_X_train, all_y_train = build_splits(ids_file="split_21-06-04_13-07_0-9668.json",
    #                                                      embeddings_file="../data/baseline_embeddings_binding_sites.h5",
    #                                                       binding_file='../data/binding_residues.txt')
