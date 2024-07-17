import csv
from typing import List
import logging
import random

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class MonostyleDataset(Dataset):
    """
    Mono-style dataset
    """

    def __init__(
        self, 
        dataset_format: str,
        dataset_path: str = None,
        sentences_list: List[str] = None,
        text_column_name: str = None,
        separator: str = None,
        style: str = None,
        max_dataset_samples: int = None,
        ):
        super(MonostyleDataset, self).__init__()

        self.allowed_dataset_formats = ["list", "csv", "line_file"]

        if dataset_format not in self.allowed_dataset_formats:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} is not supported.")

        self.dataset_format = dataset_format
        
        # Checking for list
        if self.dataset_format == "list" and sentences_list is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'sentences_list' is not provided.")
        elif self.dataset_format == "list":
            self.data = self.sentences_list

        # Checking for csv
        if self.dataset_format in ["csv"] and dataset_path is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'dataset_path' is not provided.")
        elif self.dataset_format in ["csv"] and text_column_name is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'text_column_name' is not provided.")
        elif self.dataset_format in ["csv"] and separator is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'separator' is not provided.")
        
        # Checking for line_file
        if self.dataset_format in ["line_file"] and dataset_path is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'dataset_path' is not provided.")
        elif self.dataset_format in ["line_file"] and separator is None:
            raise Exception(f"MonostyleDataset: provided dataset format {dataset_format} but required argument 'separator' is not provided.")

        self.dataset_format = dataset_format
        self.dataset_path = dataset_path
        self.text_column_name = text_column_name
        self.separator = separator
        self.style = style
        self.max_dataset_samples = max_dataset_samples

        self.load_data()

    def _load_data_csv(self):
        df = pd.read_csv(self.dataset_path, sep=self.separator, header=None)
        df.dropna(inplace=True)
        self.data = df[self.text_column_name].tolist()
        logging.debug(f"MonostyleDataset, load_data: parsed {len(self.data)} examples")
    
    def _load_data_line_file(self):
        with open(self.dataset_path) as input_file:
            self.data = input_file.read()
            self.data = self.data.split(self.separator)
        logging.debug(f"MonostyleDataset, load_data: parsed {len(self.data)} examples")   

    def load_data(self, SEED=42):

        if self.dataset_format == "csv":
            self._load_data_csv()
        elif self.dataset_format == "line_file":
            self._load_data_line_file()
        elif self.dataset_format == "list":
            logging.debug(f"MonostyleDataset, load_data: data already loaded, {len(self.data)} examples")
        else:
            raise Exception(f"MonostyleDataset, load_data: the {self.dataset_format} is not supported. Please use one of the following {self.allowed_dataset_formats}")

        random.seed(SEED)
        if self.max_dataset_samples is not None and self.max_dataset_samples < len(self.data):
            ix = random.sample(range(0, len(self.data)), self.max_dataset_samples)
            self.data = np.array(self.data)[ix].tolist()
        random.shuffle(self.data)
    
    def reduce_data(self, n_samples):
        self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ParallelRefDataset(Dataset):
    """
    ParallelRefDataset dataset
    """

    def __init__(
        self, 
        dataset_format: str,
        dataset_path_src: str = None,
        dataset_path_ref: str = None,
        n_ref: int = None,
        sentences_list_src: List[str] = None,
        sentences_list_ref: List[str] = None,
        text_column_name_src: str = None,
        text_column_name_ref: str = None,
        separator_src: str = None,
        separator_ref: str = None,
        style_src: str = None,
        style_ref: str = None,
        max_dataset_samples: int = None,
        ):
        super(ParallelRefDataset, self).__init__()

        self.allowed_dataset_formats = ["list", "csv", "line_file"]

        if dataset_format not in self.allowed_dataset_formats:
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} is not supported.")

        self.dataset_format = dataset_format
        
        # Checking for list
        if self.dataset_format == "list" and (sentences_list_src is None or sentences_list_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required arguments 'sentences_list_a' or 'sentences_list_b' is not provided.")
        elif self.dataset_format == "list":
            self.data_src = sentences_list_src
            self.data_ref = sentences_list_ref
            assert len(self.data_src) == len(self.data_ref)
        
        # Checking for csv
        if self.dataset_format in ["csv"] and (dataset_path_src is None or dataset_path_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'dataset_path_a' or 'dataset_path_b' is not provided.")
        elif self.dataset_format in ["csv"] and (text_column_name_src is None or text_column_name_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'text_column_name_a' or 'text_column_name_b' is not provided.")
        elif self.dataset_format in ["csv"] and (separator_src is None or separator_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'separator_a' or 'separator_b' is not provided.")
        
        # Checking for line_file
        if self.dataset_format in ["line_file"] and (dataset_path_src is None or dataset_path_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'dataset_path_a' or 'dataset_path_b' is not provided.")
        elif self.dataset_format in ["line_file"] and (separator_src is None or separator_ref is None):
            raise Exception(f"ParallelRefDataset: provided dataset format {dataset_format} but required argument 'separator_a' or 'separator_b' is not provided.")

        self.dataset_format = dataset_format
        self.n_ref = n_ref
        self.dataset_path_src = dataset_path_src
        self.dataset_path_ref = dataset_path_ref
        self.text_column_name_src = text_column_name_src
        self.text_column_name_ref = text_column_name_ref
        self.separator_src = separator_src
        self.separator_ref = separator_ref
        self.style_src = style_src
        self.style_ref = style_ref
        self.max_dataset_samples = max_dataset_samples

        self.load_data()

    def _load_data_csv(self):
        df_src = pd.read_csv(self.dataset_path_src, sep=self.separator_src)
        df_ref = pd.read_csv(self.dataset_path_ref, sep=self.separator_ref)
        df_src.dropna(inplace=True)
        df_ref.dropna(inplace=True)
        self.data_src = df_src[self.text_column_name_src].tolist()
        column_names = [self.text_column_name_ref+str(i) for i in range(self.n_ref)]
        self.data_ref = df_ref[column_names].to_numpy().tolist()
        assert len(self.data_src) == len(self.data_ref)
        logging.debug(f"ParallelRefDataset, load_data: parsed {len(self.data_src)} examples")

    def _load_data_line_file(self):
        with open(self.dataset_path_src) as input_file:
            self.data_src = input_file.read()
            self.data_src = self.data_src.split(self.separator_src)
        if self.style_src in ['neg', 'informal']: ref_tag = '0'
        elif self.style_src in ['pos', 'formal']: ref_tag = '1'
        self.data_ref = [[] for _ in range(len(self.data_src))]
        for i in range(self.n_ref):
            ref_path = self.dataset_path_ref+f'reference{i}.{ref_tag}.txt'
            with open(ref_path) as input_file:        
                cur_ref = input_file.read()
                cur_ref = cur_ref.split(self.separator_ref)
            for i, ref in enumerate(cur_ref):
                self.data_ref[i].append(ref)
        assert len(self.data_src) == len(self.data_ref)
        logging.debug(f"ParallelRefDataset, load_data: parsed {len(self.data_src)} examples")

    def load_data(self, SEED=42):

        if self.dataset_format == "csv":
            self._load_data_csv()
        elif self.dataset_format == "line_file":
            self._load_data_line_file()
        elif self.dataset_format == "list":
            logging.debug(f"ParallelRefDataset, load_data: data already loaded, {len(self.data_src)} examples")
        else:
            raise Exception(f"ParallelRefDataset, load_data: the {self.dataset_format} is not supported. Please use one of the following {self.allowed_dataset_formats}")
        
        if self.max_dataset_samples is not None and self.max_dataset_samples < len(self.data_src):
            random.seed(SEED)
            ix = random.sample(range(0, len(self.data_src)), self.max_dataset_samples)
            self.data_src = np.array(self.data_src)[ix].tolist()
            self.data_ref = np.array(self.data_ref)[ix].tolist()

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, idx):
        return self.data_src[idx], self.data_ref[idx]

    @staticmethod
    def customCollate(batch):
        src_list = []
        ref_list = []
        for t in batch:
            src_list.append(t[0])
            ref_list.append(t[1])
        src_list = tuple(src_list)
        return [src_list, ref_list]