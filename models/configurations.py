import json
import torch

class TaskConfig:
    def __init__(
        self, 
        subject_model_name: str, 
        input_corpus_dataset_name: str, 
        input_corpus_datapoint_num: int, 
        context_window: int,
        max_bytes_per_file: int=5 * 1024 * 1024 * 1024, # 5G
        normalized_every_vector: bool=False,
        append_msg: str=None,
    ):
        self.subject_model_name: str = subject_model_name
        self.input_corpus_dataset_name: str = input_corpus_dataset_name
        self.input_corpus_datapoint_num: int = input_corpus_datapoint_num
        self.context_window: int = context_window
        self.max_bytes_per_file: int = max_bytes_per_file
        self.normalized_every_vector: bool = normalized_every_vector
        self.append_msg: str = append_msg

    def task_name(self) -> str:
        task_name = f'{self.subject_model_name}_{self.input_corpus_dataset_name}_{self.input_corpus_datapoint_num}_{self.context_window}'

        if task_name.find("/") != -1:
            task_name = task_name.replace("/", "-")

        if self.append_msg is not None:
            task_name += f'_{self.append_msg}'
        
        return task_name

class TrainConfig:
    def __init__(
        self,
        batch_size: int=1024,
        lr: float=0.01,
        n_epochs: int=1000,
        test_size: float=0.2,
        random_state: int=0,
        sparse_penalty: float=0.0
    ):
        self.batch_size: int = batch_size
        self.lr: float = lr
        self.n_epochs: int = n_epochs
        self.test_size: float = test_size
        self.random_state: int = random_state
        self.sparse_penalty: float = sparse_penalty
    
class TrainResult:
    def __init__(
        self, 
        task_config: TaskConfig, 
        train_loss: list[float], 
        test_accuracy: list[float],
        confusion_matrix: list[torch.Tensor]
    ):
        self.task_config: TaskConfig = task_config
        self.train_loss: list[float] = train_loss
        self.test_accuracy: list[float] = test_accuracy
        self.confusion_matrix: list[torch.Tensor] = confusion_matrix

    def result_save(self):
        with open(f'./result/{self.task_config.task_name()}.json', 'w') as f:
            json.dump({
                'task_name': self.task_config.task_name(),
                'train_loss': self.train_loss,
                'test_accuracy': self.test_accuracy,
                'confusion_matrix': [cm.tolist() for cm in self.confusion_matrix]
            }, f)

class DatasetConfig:
    def __init__(
        self, 
        task_config: TaskConfig,
        test_size: float=0.2,
        tolerance: int=0,
    ):
        self.task_config: TaskConfig = task_config
        self.test_size: float = test_size
        self.tolerance: int = tolerance