import torch
from sympy.codegen.cnodes import static
from torch import nn
from transformers import BertTokenizerFast, BertModel


class EmbeddingModel(nn.Module):

    def __init__(
        self,
        tokenizer: BertTokenizerFast, model: BertModel,
        window_size: int, window_stride: int,
        device: torch.device
    ):
        super().__init__()
        self._tokenizer: BertTokenizerFast = tokenizer
        self._model: BertModel = model
        self._window_size = window_size
        self._window_stride = window_stride

        for param in self._model.parameters():
            param.requires_grad = False

        self.device = device



    @staticmethod
    def _handle_overflowing_length(x, overflow_mapping):
        sentence_embeddings_list = []
        for sentence_number in set(overflow_mapping):
            first_window_index = overflow_mapping.index(sentence_number)
            windows_amount = overflow_mapping.count(sentence_number)
            sentence_embeddings_list.append(
                x[first_window_index: first_window_index + windows_amount].mean(dim=0)
            )
        return torch.stack(sentence_embeddings_list, dim=0)


    def forward(self, x):
        x = self._tokenizer(
            x,
            truncation=True,
            max_length=self._window_size, stride=self._window_stride,
            return_tensors="pt", return_overflowing_tokens=True
        )

        for key in x.keys():
            x[key] = x[key].to(self.device)
        overflow_mapping = x.pop("overflow_to_sample_mapping").numpy().tolist()

        x = self._model(**x)
        x = x.last_hidden_state.mean(dim=1)

        return self._handle_overflowing_length(x, overflow_mapping)




if __name__ == "__main__":
    device = torch.device("cuda")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased")

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False


    text = "Hi, niggers. My name is CJ. I'm from Groove street, San Andreas, Los Santos. Groove street families for life"
    sentences = [x.rstrip() for x in text.split(".")]
    print(sentences, len(sentences))

    temp = tokenizer(
        sentences, truncation=True, max_length=7, stride=3, return_overflowing_tokens=True, return_tensors='pt', padding="max_length"
    )
    print(temp)

    overflow_to_sample_mapping = temp.pop("overflow_to_sample_mapping").numpy().tolist()

    for key in temp.keys():
        temp[key] = temp[key].to(device)


    output = model(**temp)
    print(output.last_hidden_state.shape)

    output = output.last_hidden_state.mean(dim=1)
    print(output.shape)

    sentence_embeddings = []
    for sentence_number in range(len(sentences)):
        start = overflow_to_sample_mapping.index(sentence_number)
        amount = overflow_to_sample_mapping.count(sentence_number)
        print(output[start:start+amount])
        sentence_embeddings.append(
            output[start:start+amount].mean(dim=0)
        )
    output = torch.stack(sentence_embeddings, dim=0)

    print(output.shape)
    print(output)
    #print(temp.input_ids)
    #temp.pop("overflow_to_sample_mapping")


