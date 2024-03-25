import os
import json
import torch
from tqdm import tqdm
from typing import Any
from models.model_subject import ModelSubject
from models.configurations import TaskConfig

class EmbeddingDataset:
    """
    Used for saving the embeddings from subject model

    A dataset represented with this class will be saved in the `./dataset` folder
    """
    def __init__(
        self, 
        model_ref: ModelSubject, 
        cfg: TaskConfig,
    ):
        dataset_name = cfg.task_name()

        # assertions
        dataset_path = f'./dataset/{dataset_name}'
        if os.path.exists(dataset_path):
            if len(os.listdir(dataset_path)) != 0:
                raise FileExistsError(f'{dataset_path} already exists')
        else:
            os.mkdir(dataset_path)

        # hyperparameters of this dataset
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.n_embed = model_ref.n_embed
        self.n_layer = model_ref.n_layer
        self.model_name = model_ref.model_name
        self.context_window = model_ref.context_window
        self.device = model_ref.device

        # `datapoint_num` is the number of input corpus
        # so if we have 1000 input corpus, then we have [1000 * `context_window` * `n_layer`] vectors
        # each vector has [`n_embed` + P] elements
        #    where P is the number of special informations, such as
        #    - layer index
        #    - token index
        self.datapoint_num = cfg.input_corpus_datapoint_num
        self.total_tensor_num = self.datapoint_num * self.context_window * self.n_layer
        self.real_n_embed = self.n_embed + 2

        # each tensor file should be less than `max_bytes_per_file`, so we need to calculate the number of tensors of one file
        # (one dataset will be split into multiple files)
        # those tensors not occupy one full file will also be stored separately (in the last file)
        # 
        # for example:
        #  if I need to store 500 vectors, each vector has 1000 dimensions
        #  then I need 500 * 1000 * 4 bytes = 200000 bytes
        #  if I set every tensor file has a maximum of 1M bytes
        #  then I need 200000 / 1024 / 1024 ~= 1.9 ~= 2 files
        #
        # the total number of bytes is:     total_bytes = [total_tensor_num * real_n_embed * 4]
        # the total number of files is:     file_num    = (total_bytes // max_bytes_per_file) + 1
        self.max_bytes_per_file = cfg.max_bytes_per_file
        self.file_num = ((self.total_tensor_num * self.real_n_embed * 4) // self.max_bytes_per_file) + 1
        self.tensor_num_per_file = self.total_tensor_num // self.file_num + 1

        # currently writing files state
        self.file_idx = 0
        self.tensor_idx = 0
        self.current_data = torch.zeros((self.tensor_num_per_file, self.real_n_embed), dtype=torch.float32, device=self.device)

    def _dump_config(self):
        # check if config already exists, if so, raise an error, else create the file
        config_path = f'{self.dataset_path}/config.json'
        if os.path.exists(config_path):
            raise FileExistsError(f'{config_path} already exists')
        
        with open(config_path, 'w') as f:
            json.dump({
                'n_embed': self.n_embed,
                'n_layer': self.n_layer,
                'real_n_embed': self.real_n_embed,
                'model_name': self.model_name,
                'context_window': self.context_window,
                'file_num': self.file_num,
                'tensor_num_per_file': self.tensor_num_per_file,
                'max_bytes_per_file': self.max_bytes_per_file,
            }, f)
        
    def update(self, container: Any):
        # container is a tensor shaped [N, self.real_n_embed]
        # our program guarantees that N <= self.tensor_num_per_file,
        # and from start of the process, the idx file is not full

        assert container.shape[1] == self.real_n_embed, 'The container should have the same number of embeddings as the dataset'

        # how many tensors can be inserted into the current file
        vacant_num = self.tensor_num_per_file - self.tensor_idx

        # if the length of the container is leq to the remaining space of the current page
        # then we can insert it directly
        if container.shape[0] <= vacant_num:
            self.current_data[self.tensor_idx:self.tensor_idx + container.shape[0]] = container
            self.tensor_idx += container.shape[0]
        # if the length of the container is greater than the remaining space of the current page
        # then we need to fill the current page first, and then recursively call update for the remaining
        else:
            self.current_data[self.tensor_idx:] = container[:vacant_num]
            self.tensor_idx += vacant_num
            self._save_current_data_and_update_idxs_and_data()
            self.update(container[vacant_num:])

    def _save_current_data_and_update_idxs_and_data(self):
        if self.tensor_idx == 0:
            print(f"Stop: you're trying to save an empty tensor {self.dataset_name} {self.file_idx}")
            return

        # if this page is the last, it maybe not full
        if self.tensor_idx != self.tensor_num_per_file:
            self.current_data = self.current_data[:self.tensor_idx]
        
        torch.save(self.current_data, f'{self.dataset_path}/tensor_{self.file_idx}.pt')
        
        self.current_data = torch.zeros((self.tensor_num_per_file, self.real_n_embed), dtype=torch.float32, device=self.device)
        self.file_idx += 1
        self.tensor_idx = 0

    def finish(self):
        # call this because the last page may not be full, so we need to save it manually
        self._save_current_data_and_update_idxs_and_data()
        self._dump_config()
        print(f'{self.dataset_path} collection finished')
        
class LoadEmbeddingDataset:
    """
    Load the embeddings from the dataset, for interpret models
    """
    def __init__(self, cfg: TaskConfig, check_completeness: bool=True):
        self.legal = False
        self.dataset_name = cfg.task_name()
        self.dataset_path = f'./dataset/{self.dataset_name}'
        try:
            if check_completeness:
                self.check_completeness()
            else:
                print("warning: not checking the completeness of the dataset")
            self.legal = True
        except:
            raise ValueError(f'{self.dataset_path} is not complete')
        
        if self.legal:
            self.config_path = f'{self.dataset_path}/config.json'
            with open(self.config_path, 'r') as f:
                self.__dict__.update(json.load(f))
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= self.file_num:
            raise IndexError(f'idx should be less than {self.file_num}')
        return torch.load(f'{self.dataset_path}/tensor_{idx}.pt', map_location='cpu')

    def get_test_set_from_idx(self, idx: int, test_size: float=0.2) -> torch.Tensor:
        tensor = self[idx]
        tensor = tensor[torch.randperm(tensor.shape[0])]
        return tensor[:int(tensor.shape[0] * test_size), :]

    def check_completeness(self, hard_check: bool=False) -> bool:
        print(f'Checking {self.dataset_path}...')

        config_path = f'{self.dataset_path}/config.json'

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'{config_path} not found')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        file_num = config['file_num']
        tensor_num_per_file = config['tensor_num_per_file']

        normal_tensor_shape = None
        last_tensor_shape = None

        print(f'file_num: {file_num}, tensor_num_per_file: {tensor_num_per_file}')

        tensor_path = lambda idx: f'{self.dataset_path}/tensor_{idx}.pt'

        for i in tqdm(range(file_num), desc='Checking files'):
            if not os.path.exists(tensor_path(i)):
                raise FileNotFoundError(f'{tensor_path(i)} not found')

            if hard_check:
                tensor = torch.load(tensor_path(i))
                if i < file_num - 1:
                    if not normal_tensor_shape:
                        normal_tensor_shape = tensor.shape
                    if tensor.shape[0] != tensor_num_per_file:
                        raise ValueError(f'{tensor_path(i)} has wrong shape')
                else:
                    if not last_tensor_shape:
                        last_tensor_shape = tensor.shape

        if hard_check:
            print(f'normal_tensor_shape: {normal_tensor_shape}, last_tensor_shape: {last_tensor_shape}')

        print(f'{self.dataset_path} is complete')

    