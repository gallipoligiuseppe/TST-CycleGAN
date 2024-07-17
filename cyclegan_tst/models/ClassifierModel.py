from typing import List, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from torch import nn, Tensor


class ClassifierModel(nn.Module):
    
    def __init__(
        self,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        ):
        super(ClassifierModel, self).__init__()

        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding

        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")
        self.model.eval()
    
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
