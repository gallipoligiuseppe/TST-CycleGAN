from typing import List, Optional, Tuple, Union
from torch import nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

class GeneratorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        ):
        super(GeneratorModel, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        
        if pretrained_path is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
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
        target_sentences: List[str] = None,
        device = None,
        ):

        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")

        if target_sentences is not None:
            with self.tokenizer.as_target_tokenizer():
                target = self.tokenizer(target_sentences,
                    truncation=self.truncation, 
                    padding=self.padding, 
                    max_length=self.max_seq_length,
                    return_tensors="pt")

            labels = target["input_ids"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_supervised = self.model(**inputs, labels=labels)
        
        inputs = inputs.to(device)
        output = self.model.generate(**inputs, max_length=self.max_seq_length)

        with self.tokenizer.as_target_tokenizer():
            transferred_sentences = [self.tokenizer.decode(t, skip_special_tokens=True) for t in output]

        if target_sentences is not None:
            return output, transferred_sentences, output_supervised.loss
        else:
            return output, transferred_sentences

    def transfer(
        self,
        sentences: List[str],
        device = None
        ):

        inputs = self.tokenizer(sentences, 
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")
                
        inputs = inputs.to(device)
        output = self.model.generate(**inputs, max_length=self.max_seq_length)
        with self.tokenizer.as_target_tokenizer(): 
            transferred_sentences = [self.tokenizer.decode(t, skip_special_tokens=True) for t in output]

        return transferred_sentences
    
    def save_model(
        self, 
        path: Union[str]
        ):

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")
