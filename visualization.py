from matplotlib import pyplot as plt
from embeddings import LoadEmbeddingDataset
from tqdm import tqdm
import numpy as np
import seaborn as sns
import torch

def draw_acc_loss(
        acc_array: list[float], 
        loss_array: list[float], 
        title: str,
    ) -> None:

    acc_array = [acc * 100 for acc in acc_array]
    plt.figure(figsize=(3,3))
    plt.xlabel('Epoch Process')
    plt.ylabel('Loss')
    plt.plot(loss_array, 'skyblue')
    plt.twinx()
    plt.ylabel('Accuracy(%)')
    plt.plot(acc_array, 'tomato')
    plt.title(title)
    plt.show()
    plt.close()

def draw_group(
        group: dict,
        x_label: str,
        y_label: str,
        title: str,
    ):
    """
    group: {
        'father_name1' : {
            'x_label': list,
            'y_label': list,
        }
        ...
    }
    """
    plt.figure(figsize=(3 * len(group.keys()),3))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    for father_name, data in group.items():
        plt.plot(data['x_label'], data['y_label'], label=father_name)
        plt.scatter(data['x_label'], data['y_label'])
    plt.legend()
    plt.show()
    plt.close()

def draw_dimensional_importance(
        LR_model: torch.nn.Module,
        max_num: int = 10,
        subfig_size: int = 5,
        ncols: int = 5,
    ):
    """
    Visualize each dimension's importance in the logistic regression model
    """
    weights = LR_model.weight.detach().cpu().numpy()
    # print(weights.shape) # (37, 1280)
    subfig_num = weights.shape[0]
    nrows = subfig_num // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*subfig_size, nrows*subfig_size))
    for i in range(subfig_num):
        row = weights[i]
        top_indices = np.argsort(np.abs(row))[-max_num:][::-1]
        top_values = row[top_indices]
        top_percentages = top_values / np.sum(np.abs(row)) * 100

        ax = axs[i // ncols, i % ncols]
        colors = ['tomato' if v > 0 else 'skyblue' for v in top_values]
        ax.bar(range(max_num), top_percentages, color=colors)
        ax.set_xticks(range(max_num))
        ax.set_xticklabels(top_indices)
        ax.set_title(f"Predicted as Layer {i}")
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Percentage of Total Weight (%)')

    for i in range(subfig_num, nrows*ncols):
        fig.delaxes(axs[i // ncols, i % ncols])

    plt.tight_layout()
    plt.show()
    plt.close()

def draw_cumulative_percentages(
        LR_model: torch.nn.Module,
        subfig_size: int = 5,
        ncols: int = 5,
    ):
    """
    Visualize cumulative curve of dimensions
    """
    weights = np.abs(LR_model.weight.detach().cpu().numpy())
    subfig_num = weights.shape[0]
    nrows = subfig_num // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*subfig_size, nrows*subfig_size))
    for i in range(subfig_num):
        row = weights[i]
        top_indices = np.argsort(np.abs(row))[::-1]
        top_values = row[top_indices]
        top_percentages = top_values / np.sum(np.abs(row)) * 100
        cumulative_percentages = np.cumsum(top_percentages)

        ax = axs[i // ncols, i % ncols]
        ax.plot(cumulative_percentages)
        ax.set_title(f"Predicted as Layer {i}")
        ax.set_xlabel('Top Dimensions')
        ax.set_ylabel('Cumulative Percentage (%)')

        # 50%
        ax.axhline(y=50, color='tomato', linestyle='--')
        half_index = np.argwhere(np.diff(np.sign(cumulative_percentages - 50)))[0]
        ax.axvline(x=half_index, color='tomato', linestyle='--')
        ax.annotate(
            f"x={half_index[0]}", 
            xy=(half_index, cumulative_percentages[half_index]), 
        )

        # 90%
        ax.axhline(y=90, color='skyblue', linestyle='--')
        ninety_index = np.argwhere(np.diff(np.sign(cumulative_percentages - 90)))[0]
        ax.axvline(x=ninety_index, color='skyblue', linestyle='--')
        ax.annotate(
            f"x={ninety_index[0]}", 
            xy=(ninety_index, cumulative_percentages[ninety_index]), 
        )

    for i in range(subfig_num, nrows*ncols):
        fig.delaxes(axs[i // ncols, i % ncols])

    plt.tight_layout()
    plt.show()
    plt.close()

def draw_top_predicted(
        test_dataset_name: str,
        tokenizer=None,
        test_rate: float = 0.2,
        top_num: int = 50,
    ):

    """
    Find out what tokens are most correctly predicted by the logistic regression model

    Since the related result was not computed during the training process, to visualize we should test again
    - During training process we didn't record what tensors are used for training and testing, so this test may suffer from the problem of data leakage
    - If caller doesnt provide tokenizer, we will use token index
    - All tensors will be stored on GPU, big configurations may cause OOM
    """
    dataset = LoadEmbeddingDataset(test_dataset_name, check_completeness=False)
    LR_model = torch.load(f"./result/IM_{test_dataset_name}.model").cuda()
    
    X_test = []
    Y_test = []
    token_idx_test = []

    # build test set
    for page in tqdm(range(dataset.file_num), desc='Building test set'):
        this_page = dataset[page].cuda()
        indices = torch.randperm(this_page.shape[0])
        this_page = this_page[indices]
        # split with test_rate
        split_point = int(this_page.shape[0] * test_rate)
        X_test.append(this_page[:split_point, :-2])
        Y_test.append(this_page[:split_point, -2])
        token_idx_test.append(this_page[:split_point, -1])

    tidx_correct = torch.zeros(tokenizer.vocab_size if tokenizer else 100000, dtype=torch.long).cuda()
    tidx_appear = torch.zeros(tokenizer.vocab_size if tokenizer else 100000, dtype=torch.long).cuda()

    for i in tqdm(range(len(X_test)), desc='Testing'):
        X = X_test[i]
        Y = Y_test[i]
        token_idx = token_idx_test[i].long()
        predict = LR_model(X).argmax(dim=1)
        correct = predict == Y
        correct_idx = token_idx[correct]
        for idx in correct_idx:
            tidx_correct[idx] += 1

        for idx in token_idx:
            tidx_appear[idx] += 1
    
    # sort and draw
    tidx_correct = tidx_correct.cpu()
    top_indices = tidx_correct.argsort(descending=True)[:top_num]
    top_values = tidx_correct[top_indices]

    plt.figure(figsize=(top_num//2,3))
    plt.xlabel('Token')
    plt.ylabel('Correct Prediction Times')
    if tokenizer:
        plt.xticks(range(top_num), [tokenizer.decode([i]) for i in top_indices], rotation=45)
    else:
        plt.xticks(range(top_num), [i.item() for i in top_indices], rotation=45)
    for i in range(top_num):
        plt.text(i, top_values[i], str(top_values[i].item()), ha='center', va='bottom')
    plt.bar(range(top_num), top_values, width=0.5)
    plt.show()
    plt.close()
    
def draw_dataset_token_coverance(
        dataset_name: str,
        tokenizer=None,
        ncols: int = 500,
    ):
    """
    Draw the token coverance of the dataset
    """
    dataset = LoadEmbeddingDataset(dataset_name, check_completeness=False)
    n_vocab = tokenizer.vocab_size if tokenizer else 100000
    # if n_vocab cannot be divided by ncols, we will add some padding
    if n_vocab % ncols != 0:
        n_vocab += ncols - n_vocab % ncols
    token_idx_appear = torch.zeros(n_vocab, dtype=torch.long).cuda()

    for page in tqdm(range(dataset.file_num), desc='Loading dataset'):
        this_page = dataset[page].cuda()
        token_idx = this_page[:, -1].long()
        for idx in token_idx:
            token_idx_appear[idx] += 1

    token_idx_appear = token_idx_appear.cpu()
    zero_indices = token_idx_appear == 0
    nonzero_indices = token_idx_appear != 0
    token_idx_appear[zero_indices] = 0
    token_idx_appear[nonzero_indices] = 1
    print("nonzero token number:", torch.sum(token_idx_appear).item())
    
    token_idx_appear = token_idx_appear.view(-1, ncols).numpy()


    # draw token_idx_appear as heatmap
    plt.figure(figsize=(ncols//30, (n_vocab//ncols)//30))
    sns.heatmap(token_idx_appear, cmap='YlGnBu', cbar=True)
    plt.show()
    plt.close()

def draw_confusion_matrix(
        confusion_matrix: list[torch.Tensor],
        title: str,
        ncols: int = 5,
        subfig_size: int = 5,
        hide_equal: bool = False,
    ):
    """
    Visualize the confusion matrix during training
    """
    subfig_num = len(confusion_matrix)
    nrows = subfig_num // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*subfig_size, nrows*subfig_size))
    for i in range(subfig_num):
        ax = axs[i // ncols, i % ncols]
        matrix = confusion_matrix[i].cpu().numpy()
        if hide_equal:
            matrix = matrix - np.diag(np.diag(matrix))
        sns.heatmap(matrix, ax=ax, cmap='YlGnBu', cbar=True)
        ax.set_title(f"Training Checkpoint {i}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')

    for i in range(subfig_num, nrows*ncols):
        fig.delaxes(axs[i // ncols, i % ncols])

    plt.tight_layout()
    plt.show()
    plt.close()

def draw_single_confusion_matrix(
        confusion_matrix: torch.Tensor,
        title: str,
        fig_size: int = 5,
        hide_equal: bool = False,
    ):
    """
    Visualize the confusion matrix
    """
    plt.figure(figsize=(fig_size, fig_size))
    matrix = confusion_matrix.cpu().numpy()
    if hide_equal:
        matrix = matrix - np.diag(np.diag(matrix))
    sns.heatmap(matrix, cmap='YlGnBu', cbar=True)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')

    plt.show()
    plt.close()