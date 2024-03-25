from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.configurations import TaskConfig
from typing import Any
import functools
import torch
import re

class ModelSubject:
    """
    An abstract class representing the model to be interpreted.
    This class is designed to collect embeddings from different layers.
    """
    
    def __init__(self, cfg: TaskConfig) -> None:
        """
        Subclasses should call this method first.
        """
        self.model_name: str = cfg.subject_model_name
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.context_window: int = cfg.context_window

    def register_forward_hook(self, container: Any) -> None:
        """
        Registers a forward hook to the model.
        This method should be implemented in subclasses.

        Args:
            container (Any): A container to collect outputs from the forward hook.
        """
        raise NotImplementedError("You should implement this method in a subclass.")

    def inference(self, input_corpus: str) -> Any:
        """
        Performs inference on the model and collects embeddings from different layers.
        This method should be implemented in subclasses.

        Args:
            input_corpus (str): The input text to be processed.

        Returns:
            Any: The container with the collected embeddings.
        """
        raise NotImplementedError("You should implement this method in a subclass.")

    def _encode_input(self, input_text: str) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The encoded input text, with shape [1, n_token].
        """
        return self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

    def _decode_output(self, output: torch.Tensor) -> str:
        """
        Decodes the output tensor using the tokenizer.
        """
        return self.tokenizer.decode(output[0])
    
    def _check_input_length(self, input_text: str, number: int) -> bool:
        """
        Check if the input text exceeds or is equal to the maximum token length.
        """
        return self._encode_input(input_text).shape[1] >= number


class ModelSubject_GPT2(ModelSubject):
    def __init__(self, cfg: TaskConfig) -> None:
        """
        Initializes the GPT2 model.
        """
        assert cfg.subject_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'model_name should be one of gpt2, gpt2-medium, gpt2-large, gpt2-xl'
        super().__init__(cfg)
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.n_layer: int = self.model.config.n_layer
        self.n_embed: int = self.model.config.n_embd
        self.handles: list = []

    def register_forward_hook(self, container: Any) -> None:
        """
        Registers a forward hook to the model.

        Args:
            container (Any): A container to collect outputs from the forward hook.
        """
        
        cw = self.context_window

        def hook(module, input, output, i, container):
            container[:, i] = output[0][0, :cw].detach()

        for i in range(self.n_layer):
            handle = self.model.transformer.h[i].register_forward_hook(
                functools.partial(hook, i=i, container=container)
            )
            self.handles.append(handle)

    def inference(self, input_corpus: str) -> Any:
        """
        Performs inference on the model and collects embeddings from different layers.
        
        After collection, each embedding will be labeled with layer number and other information.

        Args:
            input_corpus (str): The input text to be processed.

        Returns:
            Any: The container with the collected embeddings and layer numbers.
        """

        # Check if `token_num(input_corpus) >= context_window`
        # If not, the corpus cannot be processed
        cw = self.context_window
        if not self._check_input_length(input_corpus, cw):
            return None

        input_ids = self._encode_input(input_corpus)[:, :cw]

        # shape: [cw, n_layer, n_embed]
        container = torch.zeros(cw, self.n_layer, self.n_embed, device=self.device)
        self.register_forward_hook(container)

        output = self.model(input_ids)

        for handle in self.handles:
            handle.remove()
        self.handles = []

        # Label each embedding with layer number and other information
        _ocs = container.shape
        container = container.view(_ocs[0] * _ocs[1], _ocs[2])  # [cw * n_layer, n_embed]
        layer_info = torch.arange(self.n_layer, device=self.device).repeat(cw).view(-1, 1).float()
        container = torch.cat([container, layer_info], dim=1)   # [cw * n_layer, n_embed + 1]

        # Label each embedding with token idx
        container = torch.cat([container, input_ids[0, :cw].float().repeat_interleave(self.n_layer).view(-1, 1)], dim=1)
                                                                # [cw * n_layer, n_embed + 2]

        return container
    
class ModelSubject_Toy(ModelSubject):
    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.n_layer: int = 3
        self.n_embed: int = 4

    def register_forward_hook(self, container: Any) -> None:
        pass

    def inference(self, input_corpus: StopIteration) -> Any:
        cw = self.context_window
        # shape: [cw, n_layer, n_embed]
        container = torch.rand(cw, self.n_layer, self.n_embed, device=self.device)
        self.register_forward_hook(container)

        _ocs = container.shape
        # shape: [cw * n_layer, n_embed]
        container = container.view(_ocs[0] * _ocs[1], _ocs[2])
        # Add layer number to each embedding
        layer_info = torch.arange(self.n_layer, device=self.device).repeat(cw).view(-1, 1).float()
        # shape: [cw * n_layer, n_embed + 1]
        container = torch.cat([container, layer_info], dim=1)

        input_ids = torch.tensor([[888, 999]], device=self.device)
        # shape: [cw * n_layer, n_embed + 2]
        container = torch.cat([container, input_ids[0, :cw].float().repeat_interleave(self.n_layer).view(-1, 1)], dim=1)

        return container

class ModelSubject_Llama2(ModelSubject):
    def __init__(self, cfg: TaskConfig) -> None:
        """
        Initializes the Llama2 model, especially for Llama2-7b-chat.

        NOTE:
            - the tokenizer will add a `<s>` token at the beginning of the input text.
        """
        assert re.match(r'^meta-llama/Llama-2-(7b|13b|70b)(-chat)?-hf$', cfg.subject_model_name), f'model {cfg.subject_model_name} is not supported'
        super().__init__(cfg)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.n_layer: int = self.model.config.num_hidden_layers
        self.n_embed: int = self.model.config.hidden_size
        self.handles: list = []

    def register_forward_hook(self, container: Any) -> None:
        """
        Registers a forward hook to the model.

        Args:
            container (Any): A container to collect outputs from the forward hook.
        """

        cw = self.context_window

        def hook(module, input, output, i, container):
            # Batch obtain layer_i output embeddings of all tokens in context window
            container[:, i] = output[0][0, :cw].detach()

        for i in range(self.n_layer):
            handle = self.model._modules['model'].layers[i].register_forward_hook(
                functools.partial(hook, i=i, container=container)
            )
            self.handles.append(handle)

    def inference(self, input_corpus: str) -> Any:
        """
        Performs inference on the model and collects embeddings from different layers.
        
        return tensor size: [context_window * n_layer, n_embed + 2]
        """

        # Check if the input text exceeds or is equal to the maximum token length
        # If not, the corpus cannot be processed, return None
        cw = self.context_window
        if not self._check_input_length(input_corpus, cw):
            return None

        # Encode the input text and collect embeddings from different layers
        input_ids = self._encode_input(input_corpus)[:, :cw]
        # input_corpus = self._trunc_input(input_corpus, cw)
        # input_ids = self._encode_input(input_corpus)

        # shape: [cw, n_layer, n_embed]
        container = torch.zeros(cw, self.n_layer, self.n_embed, device=self.device)
        self.register_forward_hook(container)

        output = self.model(input_ids)

        for handle in self.handles:
            handle.remove()
        self.handles = []

        # Label each embedding with layer number and other information
        _ocs = container.shape
        # shape: [cw * n_layer, n_embed]
        container = container.view(_ocs[0] * _ocs[1], _ocs[2])
        # Add layer number to each embedding
        layer_info = torch.arange(self.n_layer, device=self.device).repeat(cw).view(-1, 1).float()
        # shape: [cw * n_layer, n_embed + 1]
        container = torch.cat([container, layer_info], dim=1)

        # 新：将每一行所代表的 token 的信息加入到 container 中
        # shape: [cw * n_layer, n_embed + 2]
        container = torch.cat([container, input_ids[0, :cw].float().repeat_interleave(self.n_layer).view(-1, 1)], dim=1)

        return container