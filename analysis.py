import os
import json
import torch

class TrainResultAnalysis:
    """
    Used for analyse the result of training interpret models

    `task_name` should in format of `{subject_model_name}_{input_corpus_dataset_name}_{input_corpus_datapoint_num}_{context_window}`

    For example, `gpt2_Pile_5000_10`  
    """
    def __init__(self, task_name: str):
        self.task_name = task_name

        args = task_name.split('_')
        if len(args) == 4:
            self.subject_model_name, self.input_corpus_dataset_name, self.input_corpus_datapoint_num, self.context_window = args
        elif len(args) == 5:
            self.subject_model_name, self.input_corpus_dataset_name, self.input_corpus_datapoint_num, self.context_window, self.append_msg = args

        self.interpret_model_name = f"./result/IM_{self.task_name}.model"

        # load result
        with open(f"./result/{self.task_name}.json", 'r') as f:
            result = json.load(f)
            assert self.task_name == result['task_name'], "Task name not match"
            self.train_loss: list[float] = result['train_loss']
            self.test_accuracy: list[float] = result['test_accuracy']
            if 'confusion_matrix' in result:
                self.confusion_matrix: list[torch.Tensor] = [torch.tensor(cm) for cm in result['confusion_matrix']]

        # load interpret model
        self.interpret_model = torch.load(self.interpret_model_name)

        # final acc estimate
        self.final_acc = self.test_accuracy[-1]
