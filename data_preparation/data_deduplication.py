import json
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.metrics import jaccard_score
import torch
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from argparse import ArgumentParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def checkIfSubArray(seqA, seqB, return_last_index=False):
    '''
    Args:
        seqA: Can be a string or sequence of tokens (array or list)
        seqB: ''
    Returns: check whether sequence seqB is a subset of a larger sequence seqA
    '''
    if type(seqA) == str:
        seqA = seqA.split()
    if type(seqB) == str:
        seqB = seqB.split()
    n, m = len(seqA), len(seqB)

    i, j = 0, 0
    while (i < n and j < m):
        if (seqA[i] == seqB[j]):
            i += 1
            j += 1
            if (j == m):
                if return_last_index:
                    return True, (i - m, i)
                return True
        else:
            i = i - j + 1
            j = 0
    if return_last_index:
        return False, None
    return False

def jaccard_similarity(doc1: str, doc2: str):
    '''
    Returns:computes and returns the Jaccard index score between doc1 and doc2 which are both strings
    '''
    set1, set2 = set(doc1.split()), set(doc2.split())
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    return intersection / union


def searchStringSimilarity(index, query, corpus):
    '''
    Args:
    Returns sentences within a corpus ranked based on their similarity (computed using jaccard) with a query sentence
    '''
    query = np.array([query.split()]).reshape(-1,1)
    sents_scores = []
    query_index = index
    for sent in corpus:
        query_index += 1
        sent = np.array([sent.split()]).reshape(-1,1)
        jaccard_index = jaccard_score(query, sent)
        sents_scores.append((query_index, jaccard_index, sent))
    ranked_results = sorted(sents_scores, key=lambda x:x[1])
    return ranked_results

if __name__ == "__main__":
    par = ArgumentParser()
    par.add_argument("--file", type=str, help="path to the dataset file")
    args = par.parse_args()

    dataset = pd.read_csv(args.file, usecols=["ids", "ClientID", "EnterDatetime", "NoteType", "NoteTypeDesc", "NoteTextCleaned", "NoteTextDeidentified"], index_col=0)

    #drop duplcates based on NoteText
    dataset = dataset.drop_duplicates(subset=["NoteTextCleaned"])

    de_duplicated_dataset = dataset.copy()

    de_duplicated_data = {}
    duplicate_ids_list = []
    clients = list(set(dataset['ClientID'].tolist()))

    for client in tqdm(clients):
        client_dataset = dataset[dataset['ClientID'] == client].copy()
        # print(client_dataset)
        client_dataset["NoteTextLen"] = client_dataset["NoteTextCleaned"].apply(lambda x: len(x.split()))
        client_dataset = client_dataset.sort_values(by="NoteTextLen", ascending=False, ignore_index=True)
        duplicate_ids, duplicate_jaccard = {}, {}
        client_dataset_corpus = np.array(client_dataset[["ids", "NoteTextCleaned"]]).tolist()
        for i, note_entry in enumerate(client_dataset_corpus):
            # print(note_entry[1], "\n")
            note_entry_id = note_entry[0]
            duplicate_ids[note_entry_id] = []
            duplicate_jaccard[note_entry_id] = []
            if note_entry_id not in duplicate_ids_list:
                for j, compared_entry in enumerate(client_dataset_corpus[i + 1:]):
                    compared_entry_id = compared_entry[0]
                    similar = checkIfSubArray(note_entry[1], compared_entry[1])
                    if similar:
                        duplicate_ids[note_entry_id].append(compared_entry_id)
                        duplicate_ids_list.append(compared_entry_id)
                    else:
                        sim_score = jaccard_similarity(note_entry[1], compared_entry[1])
                        if sim_score > 0.9:
                            duplicate_jaccard[note_entry_id].append(compared_entry_id)
            else:
                pass

        de_duplicated_data[client] = duplicate_ids

    # remove duplicated data
    if not os.path.exists("de_duplicated_data_jaccard.json"):
        print("========================================================================")
        with open("de_duplicated_data_jaccard.json", "w") as dedup:
            json.dump(de_duplicated_data, dedup, indent=2)
            dedup.close()

    with open("de_duplicated_data.json") as dup:
        d = json.load(dup)
        already_removed = []
        key_ids = []
        for client in d:
            for k, v in d[client].items():
                if k not in key_ids:
                    key_ids.append(k)
                if len(v) > 0:
                    for i in v:
                        if i not in already_removed:
                            already_removed.append(i)

    if len(already_removed) > len(key_ids):
        overlap = [i for i in key_ids if i in already_removed]
    else:
        overlap = [i for i in already_removed if i in key_ids]
    print(overlap, len(key_ids), len(already_removed))

    de_duplicated_dataset = de_duplicated_dataset[~de_duplicated_dataset["ids"].isin(already_removed)]