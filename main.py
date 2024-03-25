from classifier import *
from models.configurations import *
from pipeline import pipeline

task_config = TaskConfig(
    # subject_model_name='gpt2',
    subject_model_name='meta-llama/Llama-2-13b-chat-hf',
    input_corpus_dataset_name='Pile',
    input_corpus_datapoint_num=3000,
    context_window=30,
    normalized_every_vector=False,
    # append_msg="normalized",
)

train_config = TrainConfig(
    sparse_penalty=0,
)

pipeline(
    task_cfg=task_config,
    train_cfg=train_config,
)