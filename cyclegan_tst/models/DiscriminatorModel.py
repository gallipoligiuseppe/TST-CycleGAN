from typing import List, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from torch import nn, Tensor


class DiscriminatorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        ):
        super(DiscriminatorModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding

        if pretrained_path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")
    

    def train(self):
        # Setting the model in training mode
        self.model.train()

    def eval(self):
        # Setting the model in evaluation mode
        self.model.eval()

    def forward(
        self,
        sentences: List[str],
        target_labels: Tensor,
        return_hidden: bool = False,
        device=None,
        ):

        inputs = self.tokenizer(sentences,
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")

        inputs["labels"] = target_labels
        inputs = inputs.to(device)
        output = self.model(**inputs, output_hidden_states=return_hidden)

        return output, output.loss
    
    def save_model(
        self, 
        path: Union[str]
        ):

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
