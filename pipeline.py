from classifier import *
from models.configurations import TaskConfig, TrainConfig, DatasetConfig
from models.model_subject import ModelSubject_GPT2, ModelSubject_Llama2, ModelSubject_Toy
from visualization import *
from embeddings import *
from corpusmaker import TextCorpus, Pile
from tqdm import tqdm
import os

def get_subject_model_from_config(config: TaskConfig,) -> ModelSubject:
    if config.subject_model_name.startswith('gpt2'):
        return ModelSubject_GPT2(config)
    elif config.subject_model_name.startswith('meta-llama'):
        return ModelSubject_Llama2(config)
    elif config.subject_model_name.startswith('Toy'):
        return ModelSubject_Toy(config)
    else:
        raise NotImplementedError(f"Model {config.subject_model_name} not implemented")

def get_input_corpus_from_config(config: TaskConfig) -> TextCorpus:
    if config.input_corpus_dataset_name == 'Pile':
        return Pile(
            pile_dataset_path='./dataset/Pile2-processed.json',
            num_total=config.input_corpus_datapoint_num,
        )
    else:
        raise NotImplementedError(f"Dataset {config.input_corpus_dataset_name} not implemented")

def make_embeddings_from_config(
        config: TaskConfig,
    ):
    subject_model = get_subject_model_from_config(config)
    input_corpus_dataset = get_input_corpus_from_config(config)

    embedding_dataset = EmbeddingDataset(
        model_ref=subject_model, 
        cfg=config
    )

    assert len(input_corpus_dataset.raw) >= config.input_corpus_datapoint_num, "dataset is too small to satisfy config.datapoint_num"

    for i in tqdm(range(config.input_corpus_datapoint_num)):
        text = input_corpus_dataset.raw[i]
        container = subject_model.inference(text)

        if container is None:
            continue

        if config.normalized_every_vector:
            data_part = container[:, :subject_model.n_embed]
            data_part = data_part / torch.norm(data_part, dim=1, keepdim=True)
            container = torch.cat([data_part, container[:, subject_model.n_embed:]], dim=1)

        embedding_dataset.update(container)

    embedding_dataset.finish()

    print(f'Finished making embeddings for {config.task_name()}')

def pipeline(task_cfg: TaskConfig, train_cfg: TrainConfig):
    """
    From dataset building to training

    A model and a dataset will be built from the given configuration

    Training loss & acc will be saved in `./result` folder
    """
    dataset_name = task_cfg.task_name()
    interpret_model_name = f"./result/IM_{dataset_name}.model"

    if os.path.exists(interpret_model_name):
        print(f"Skipping {dataset_name}...")
        return

    # if dataset not exists, build it
    if not os.path.exists(f"./dataset/{dataset_name}"):
        print(f"Building dataset {dataset_name}...")
        make_embeddings_from_config(task_cfg)
        print(f"Finished building dataset {dataset_name}")

    clsfer = LayerClassifier()
    dataset = LoadEmbeddingDataset(task_cfg)
    print(f"Now training {dataset_name}...")

    result: TrainResult = clsfer.train(
        dataset=dataset, 
        task_config=task_cfg,
        train_config=train_cfg,
    )

    result.result_save()
    
    clsfer.interpret_model.save(interpret_model_name)

def transferability_test(
    model_config: TaskConfig, 
    dataset_config: DatasetConfig
) -> tuple[float, torch.Tensor]:
    """
    Test the transferability of the model which is trained on another dataset
    """
    dataset = LoadEmbeddingDataset(dataset_config.task_config)
    interpret_model = torch.load(f"./result/IM_{model_config.task_name()}.model").cuda()

    hit_num, total_num = 0, 0
    conf_mat = torch.zeros(dataset.n_layer, interpret_model.weight.data.shape[0]).cuda()

    for page in tqdm(range(dataset.file_num), desc='Testing'):
        test_tensor = dataset.get_test_set_from_idx(page, test_size=dataset_config.test_size).cuda()
        X = test_tensor[:, :-2].cuda()
        Y = test_tensor[:, -2].cuda().long()
        Y_hat = torch.argmax(interpret_model(X), dim=1).long()

        hit_num += torch.sum(torch.abs(Y - Y_hat) <= dataset_config.tolerance).item()
        total_num += Y.shape[0]

        for i in range(Y.shape[0]):
            conf_mat[Y[i], Y_hat[i]] += 1

    return hit_num / total_num, conf_mat.cpu()