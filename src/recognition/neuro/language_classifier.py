# Author: wormiz
import torch
from torch import nn


def define_device():
    if torch.cuda.is_available():
        print("Running on gpu")
        return torch.device("cuda")
    else:
        print("Running on cpu")
        return torch.device("cpu")


class LanguageClassifier(nn.Module):

    @staticmethod
    def _linear_norm_dropout(input_size, output_size, device, dtype):
        linear_layer = nn.Linear(
            input_size, output_size, bias=False, device=device, dtype=dtype
        )
        batch_norm_layer = nn.BatchNorm1d(output_size, device=device, dtype=dtype)
        dropout_layer = nn.Dropout(0.5)
        return linear_layer, batch_norm_layer, dropout_layer
        

    def _init_layers(self, encoder_output_size, hidden_layers, num_classes, device, dtype):
        if hidden_layers:
            _hidden_layers = nn.ModuleList()
            linear, batch_norm, dropout = self._linear_norm_dropout(
                encoder_output_size, hidden_layers[0], device, dtype 
            )
            _hidden_layers.extend([linear, batch_norm, dropout, nn.ReLU()])

            for i in range(1, len(hidden_layers)):
                linear, batch_norm, dropout = self._linear_norm_dropout(
                    hidden_layers[i - 1], hidden_layers[i], device, dtype 
                )
                _hidden_layers.extend([linear, batch_norm, dropout, nn.ReLU()])
            
            self._hidden_network = nn.Sequential(*_hidden_layers)
            self._output_layer = nn.Linear(hidden_layers[-1], num_classes, device=device, dtype=dtype)
        else:
            self._hidden_network = None
            self._output_layer = nn.Linear(encoder_output_size, num_classes, device=device, dtype=dtype)


    def init_weights(self):
        nn.init.xavier_normal_(self._output_layer)
        for module in self._hidden_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module)


    def turn_off_encoder_grads(self):
        for encoder_param in self._encoder_model.parameters():
                encoder_param.requires_grad = False


    def __init__(
            self,
            tokenizer, encoder_model, encoder_output_size,
            window_size, window_stride,
            num_classes,
            device, layers_dtype, tokenizer_dtype,
            hidden_layers=None,
            encoder_requires_gradients=False
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._encoder_model = encoder_model.to(device)
        if not encoder_requires_gradients:
            self.turn_off_encoder_grads()
        
        self._init_layers(encoder_output_size, hidden_layers, num_classes, device, layers_dtype)

        self._device = device
        self._window_size = window_size
        self._window_stride = window_stride
        self._layers_dtype = layers_dtype
        self._tokenizer_dtype = tokenizer_dtype
        self._num_classes = num_classes


    def _tokenize_text(self, text):
        tokenized_text = self._tokenizer(
            text,
            padding="max_length", truncation=True,
            max_length=self._window_size, stride=self._window_stride,
            return_overflowing_tokens=True,
            return_tensors="pt"
        )
        tokenized_text.pop("overflow_to_sample_mapping")

        for key in tokenized_text.keys():
            tokenized_text[key] = tokenized_text[key].type(self._tokenizer_dtype).to(self._device)
        
        return tokenized_text
    

    def _process_text(self, text):
        tokenized_text = self._tokenize_text(text)
        processed_text = self._encoder_model(**tokenized_text).last_hidden_state
        processed_text = processed_text.mean(dim=1).type(self._layers_dtype)
        return processed_text


    def train_forward(self, texts_batch, return_batch_mapping=False):
        model_outputs = []
        if return_batch_mapping:
            batch_mapping_list = []
        with torch.no_grad():
            for i in range(len(texts_batch)):
                processed_text = self._process_text(texts_batch[i])
                for j in range(processed_text.shape[0]):
                    model_outputs.append(processed_text[j])
                    if return_batch_mapping:
                        batch_mapping_list.append(i)
        output = torch.stack(model_outputs, dim=0)

        if self._hidden_network:
            output = self._hidden_network(output)
        output = self._output_layer(output)

        if return_batch_mapping:
            return output, batch_mapping_list
        else:
            return output


    def forward(self, text, requieres_gradient=False):
        return self.count_probabilities(text, requieres_gradient)


    def count_probabilities(self, text, requieres_gradient=False):
        if requieres_gradient:
            processed_text = self._process_text(text)

            if self._hidden_network:
                output = self._hidden_network(processed_text)
                output = self._output_layer(output)
            else:
                output = self._output_layer(processed_text)
            output = torch.softmax(output, dim=1)
            return output
        else:
            with torch.no_grad():
                processed_text = self._process_text(text)

                if self._hidden_network:
                    output = self._hidden_network(processed_text)
                    output = self._output_layer(output)
                else:
                    output = self._output_layer(processed_text)
                output = torch.softmax(output, dim=1)
                return output


    def predict(self, text, method="mode-argmax", return_one_hot=False):
        with torch.no_grad():
            output = self.count_probabilities(text, requieres_gradient=False)

            if method.lower() == "mode-argmax":  # can make worth dicisions, when more, than one modes
                _, labels = torch.max(output, dim=1)  # but can make better dicisions, when there are a few anomalies
                label, _ = torch.mode(labels, dim=0)
            elif method.lower() == "argmax-mean":
                _, label = torch.max(output.sum(dim=0), dim=0)  # max(mean) = max(sum) because tensors have the same shape
            else:
                raise ValueError(
                    f"expected method to be mode-argmax or argmax-mean, but got {method} instead"
                )
            if return_one_hot:
                return nn.functional.one_hot(label, num_classes=self._num_classes)
            else:
                return label
            




