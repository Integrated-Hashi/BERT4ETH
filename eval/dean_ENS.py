import numpy as np
import os

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser("dean_ens")
parser.add_argument("--input_dir", type=str, default="../outputs/bert4eth_filter_epoch_50", help="the input directory of address and embedding list")
parser.add_argument("--max_cnt", type=int, default=2)


args = parser.parse_args()

def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
    Generate testing pairs based on ENS name
    :param ens_pairs:
    :param min_cnt:
    :param max_cnt:
    :param mirror:
    :return:
    """
    pairs = ens_pairs.copy()
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    all_ens_names = []
    ename2addresses = {}
    for idx, row in pairs.iterrows():
        try:
            ename2addresses[row["name"]].append(row["address"]) # note: cannot use row.name
        except:
            ename2addresses[row["name"]] = [row["address"]]
    for cnt in range(min_cnt, max_cnt + 1):
        ens_names = list(ens_counts[ens_counts == cnt].index)
        all_ens_names += ens_names
        # convert to indices
        for ename in ens_names:
            addrs = ename2addresses[ename]
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    addr1, addr2 = addrs[i], addrs[j]
                    address_pairs.append([addr1, addr2])
                    if mirror:
                        address_pairs.append([addr2, addr1])
    return address_pairs, all_ens_names

def load_embedding():
    address_input_dir = os.path.join(args.input_dir, "address.npy")
    embed_input_dir = os.path.join(args.input_dir, "embedding.npy")
    address_for_embedding = np.load(address_input_dir)
    embeddings = np.load(embed_input_dir)

    # group by embedding according to address
    address_to_embedding = {}

    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    X = np.array(np.squeeze(embedding_list))

    return X, address_list

def main():
    ens_pairs = pd.read_csv("../data/dean_all_ens_pairs.csv")
    max_ens_per_address = 1
    num_ens_for_addr = ens_pairs.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    excluded = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    ens_pairs = ens_pairs[~ens_pairs["address"].isin(excluded)]
    address_pairs, all_ens_names = generate_pairs(ens_pairs, max_cnt=args.max_cnt)

    X, address_list = load_embedding()
    print(X)

if __name__ == '__main__':
    main()
