import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from embeddings import LoadEmbeddingDataset
from models.model_interpret import *
from models.configurations import TrainConfig, TrainResult, TaskConfig

class LayerClassifier:
    """
    A class for training and testing a logistic regression model on a given dataset.
    """
    def __init__(self):
        self.interpret_model: Model_Interpret = None
        self.X_test: list[torch.Tensor] = []
        self.Y_test: list[torch.Tensor] = []
        self.X_train: list[torch.Tensor] = []
        self.Y_train: list[torch.Tensor] = []

    def train(
        self, 
        dataset: LoadEmbeddingDataset, 
        task_config: TaskConfig,
        train_config: TrainConfig,
    ) -> TrainResult:
        """
        Train the logistic regression model using the specified dataset and training parameters.

        Returns:
            training loss & test accuracy & confusion matrix during training
        """
        
        self._prepare_dataset(dataset, train_config)
        self.n_features = dataset.n_embed
        self.n_classes = dataset.n_layer
        self.interpret_model = Model_LogisticRegression(self.n_features, self.n_classes).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.interpret_model.parameters(), lr=train_config.lr)

        loss_list, acc_list, conf_mat_list = self._train_model(criterion, optimizer, train_config)
        
        return TrainResult(
            task_config=task_config,
            train_loss=loss_list,
            test_accuracy=acc_list,
            confusion_matrix=conf_mat_list
        )
    
    def _prepare_dataset(
        self, 
        dataset: LoadEmbeddingDataset, 
        config: TrainConfig,
    ) -> None:
        """
        Prepares the dataset for training and testing by splitting it into training and test sets.

        Args:
            dataset (EmbeddingDataset): The dataset to prepare.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        if dataset.legal == False:
            raise ValueError('The dataset is not legal')
        
        for i in range(dataset.file_num):
            X_train, X_test, Y_train, Y_test = train_test_split(
                dataset[i][:, :-2],
                dataset[i][:, -2].long(),
                test_size=config.test_size,
                random_state=config.random_state
            )

            # normalization
            X_train = (X_train - X_train.mean()) / X_train.std()
            X_test = (X_test - X_test.mean()) / X_test.std()

            self.X_train.append(X_train)
            self.Y_train.append(Y_train)
            self.X_test.append(X_test)
            self.Y_test.append(Y_test)
    
    def _train_model(
        self, 
        criterion, 
        optimizer, 
        config: TrainConfig,
    ) -> tuple[list[float], list[float], list[torch.Tensor]]:
        """
        Trains the model on the training dataset.

        Returns:
            training loss & test accuracy & confusion matrix during training
        """

        loss_list = []
        acc_list = []
        conf_mat_list = []
        
        for epoch in tqdm(range(config.n_epochs), desc="Training Progress"):
            # % new epoch start
            for page in range(len(self.X_train)):
                # % new page start
                X_train = self.X_train[page].cuda()
                Y_train = self.Y_train[page].cuda()
                n_samples = X_train.shape[0]

                # shuffle the data
                indices = torch.randperm(n_samples)
                X_train = X_train[indices]
                Y_train = Y_train[indices]

                for i in range(0, n_samples, config.batch_size):
                    end = min(i + config.batch_size, n_samples)
                    
                    X = X_train[i:end]
                    Y = Y_train[i:end]

                    # forward pass
                    Yhat = self.interpret_model(X)
                    loss = criterion(Yhat, Y)

                    # L1 penalty
                    if config.sparse_penalty > 0:
                        l1_reg = torch.tensor(0., requires_grad=True).cuda()
                        for param in self.interpret_model.parameters():
                            l1_reg += torch.norm(param, 1)
                        loss += config.sparse_penalty * l1_reg

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                # % new page end
            # % new epoch end
            if epoch % 10 == 0:
                loss_list.append(loss.item())
                acc_item, conf_mat = self._test_acc(cal_conf_matrix=(epoch % 100 == 0))
                acc_list.append(acc_item)
                if conf_mat is not None:
                    conf_mat_list.append(conf_mat)
                
        return loss_list, acc_list, conf_mat_list

    def _test_acc(self, cal_conf_matrix=False) -> tuple[float, torch.Tensor|None]:
        """
        Tests the logistic regression model.
        """
        with torch.no_grad():
            correct = 0
            total = 0

            if cal_conf_matrix:
                confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
            else:
                confusion_matrix = None

            for page in range(len(self.X_test)):
                X_test = self.X_test[page].cuda()
                Y_test = self.Y_test[page].cuda()
                Yhat = self.interpret_model(X_test)
                predicted = torch.argmax(Yhat, dim=1)
                # accuracy
                total += Y_test.size(0)
                correct += (predicted == Y_test).sum().item()
                # confusion matrix
                if confusion_matrix is not None:
                    for t, p in zip(Y_test.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        
        return correct / total, confusion_matrix