# Author: wormiz
import sys  # TODO remove
sys.path.append("/home/vodohleb/PycharmProjects/language-recognition")

import datetime
import torch
from torch.nn import functional as F
from sklearn import metrics
import torch.utils
import torch.utils.data
from transformers import BertTokenizerFast, BertModel 
from src.recognition.neuro.language_classifier import LanguageClassifier, define_device
from src.models.model import Language
from src.recognition.neuro.dataset import CustomDataset



class NeuroMethod:
    
    labels_mapping = {
        0: Language.ENGLISH,
        1: Language.RUSSIAN
    }

    def __init__(self, classifier_model, device, dtype, weights_filename=None):
        self._classifier: LanguageClassifier = classifier_model
        self._device = device
        self._dtype = dtype

        if weights_filename \
            and isinstance(weights_filename, str) \
                and weights_filename.endswith(".pt"):
            classifier_model.load_state_dict(torch.load(weights_filename))


    def _train_phase(
        self, optimizer, loss_function, dataloader, train_loss, train_elements_amount
    ):
        self._classifier.train()
        iters = 0
        for texts_batch, labels_batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            logits, batch_mapping = self._classifier.train_forward(
                texts_batch, return_batch_mapping=True
            )

            target_labels = []
            for index in batch_mapping:
                target_labels.append(self._one_hot(labels_batch[index]))
            target_labels = torch.as_tensor(target_labels).type(self._dtype).to(self._device)

            loss = loss_function(logits, target=target_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.detach().cpu().item()
            train_elements_amount += len(target_labels)
            iters += 1
        return train_loss / iters, train_elements_amount


    @classmethod
    def _one_hot(cls, label):
        encoded = []
        for i in range(len(cls.labels_mapping)):
            encoded.append(1 if i == label else 0)
        return encoded


    def _validation_phase(
        self, loss_function, dataloader, validation_loss, validation_elements_amount
    ):
        self._classifier.eval()
        iters = 0
        for texts_batch, labels_batch in dataloader:
            logits, batch_mapping = self._classifier.train_forward(
                texts_batch, return_batch_mapping=True
            )

            target_labels = []
            for index in batch_mapping:
                target_labels.append(self._one_hot(labels_batch[index]))
            target_labels = torch.as_tensor(target_labels).type(self._dtype).to(self._device)
            
            loss = loss_function(logits, target=target_labels)
            validation_loss += loss.detach().cpu().item()
            validation_elements_amount += len(target_labels)
            iters += 1
        return validation_loss / iters, validation_elements_amount
    
    def _save_model(self, file_to_save_model):
        print("saving model...")
        if isinstance(file_to_save_model, str) and file_to_save_model.endswith(".pt"):
            torch.save(self._classifier.state_dict(), file_to_save_model)
        else:
            raise ValueError(
                f"Expected file_to_save_model to be "
                f"filename with extention .pt, but got {file_to_save_model} instead"
            )
        print("model saved")

    def _train_validation_actions(self, optimizer, loss_function, train_loader, validation_loader):
        epoch_start = datetime.datetime.now()
        train_loss = 0.0
        train_elements_amount = 0
        validation_loss = 0.0
        validation_elements_amount = 0
        print("\tTrain")
        train_loss, train_elements_amount = self._train_phase(
            optimizer, loss_function, train_loader, train_loss, train_elements_amount
        )
        print("\tValidation")
        with torch.no_grad():
            validation_loss, validation_elements_amount = self._validation_phase(
                loss_function, validation_loader, validation_loss, validation_elements_amount
            )
        # TODO check if needed validation_loss = validation_loss / validation_elements_amount

        print(
            f"\t\tTime spent on epoch: {datetime.datetime.now() - epoch_start}\n"
            f"\t\ttrain losses: {train_loss}\n"# TODO check if needed {train_loss / train_elements_amount}\n"
            f"\t\tval losses: {validation_loss}"
        )
        return validation_loss

    def train(
        self, optimizer, loss_function, epochs, train_loader, validation_loader,
        file_to_save_best_val_model=None, file_to_save_last_model=None
    ):
        start = datetime.datetime.now()
        best_val_loss = torch.Tensor([1e10]).type(self._dtype)
        best_val_loss_epoch = 0
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")
            
            validation_loss = self._train_validation_actions(
                optimizer, loss_function, train_loader, validation_loader
            )

            if file_to_save_best_val_model and validation_loss < best_val_loss:
                print("Validation loss became better, ", end='')
                self._save_model(file_to_save_best_val_model)
                best_val_loss = validation_loss
                best_val_loss_epoch = epoch
        print(f"Time spent on training:\t{datetime.datetime.now() - start}\n"
              f"\tBest vaidation loss: {best_val_loss} on epoch {best_val_loss_epoch + 1}")
        if file_to_save_last_model:
            self._save_model(file_to_save_best_val_model)
            

    def test(self, test_loader):
        predictions = []
        real_labels = []
        for text, label in test_loader:
            real_labels.append(label)
            predictions.append(self._classifier.predict(text, method="mode-argmax").cpu())
        self.metrics(predictions, real_labels)

    
    def get_language(self, text):
        label = self._classifier.predict(text, method="mode-argmax", return_one_hot=False)
        return self.labels_mapping[label]
    

    def train_mod(self):
        self._classifier.train()
    

    def eval_mod(self):
        self._classifier.eval()

    
    @classmethod
    def metrics(cls, predictions, real_labels):
        precisions = metrics.precision_score(real_labels, predictions, average=None)
        macro_precision = 0
        print("Precision:")
        for i in range(precisions.shape[0]):
            print(f"\t{cls.labels_mapping[i].value}: {precisions[i]}")
            macro_precision += precisions[i] / precisions.shape[0]
        print(f"Macro-precision: {macro_precision}")
        
        recalls = metrics.recall_score(real_labels, predictions, average=None)
        macro_recall = 0
        print("Recall:")
        for i in range(recalls.shape[0]):
            print(f"\t{cls.labels_mapping[i].value}: {recalls[i]}")
            macro_recall += recalls[i] / recalls.shape[0]
        print(f"Macro-recall: {macro_recall}")        

        print(f"Accuracy: {metrics.accuracy_score(real_labels, predictions)}")


def auto_create_neuro(
    window_size, window_stride, num_classes,
    custom_labels_mapping=None, path_to_load_model=None, return_classifier=False
):
    device = define_device()
    layers_dtype = torch.float64
    tokenizer_dtype = torch.int64
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-uncased")
    encoder = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased")

    classifier_model = LanguageClassifier(
        tokenizer, encoder,
        window_size, window_stride, num_classes,
        device, layers_dtype, tokenizer_dtype,
        encoder_output= 768, hidden_layers=[384, 192], encoder_requires_gradients=False
    )
    neuro_method = NeuroMethod(classifier_model, device, layers_dtype, weights_filename=path_to_load_model)
    if custom_labels_mapping is not None:
        neuro_method.labels_mapping = custom_labels_mapping
    if return_classifier:
        return neuro_method, classifier_model
    else:
        return neuro_method


def load_text(path, shuffle=False):
    with open(path, 'r') as f:
        texts = json.load(f)
    if shuffle:
        random.shuffle(texts)
    return texts


def make_dataset_content(regexp, shuffle=False):
    en_texts = load_text("/home/vodohleb/PycharmProjects/huyna/en.json", shuffle=shuffle)
    ru_texts = load_text("/home/vodohleb/PycharmProjects/huyna/ru.json", shuffle=shuffle)
    texts = []
    labels = []
    en_index, ru_index = 0, 0

    for _ in range(len(en_texts) + len(ru_texts)):
        if en_index == len(en_texts):
            choice = 1
        elif ru_index == len(ru_texts):
            choice = 0
        else:
            choice = random.randint(0, 9)
        if choice % 2 == 0:
            texts.append(regexp.sub(' ', en_texts[en_index]["text"]).strip())
            labels.append(0)
            en_index += 1
        else:
            texts.append(regexp.sub(' ', ru_texts[ru_index]["text"]).strip())
            labels.append(1)
            ru_index += 1
    return texts, labels


def make_dataloaders(texts, labels):
    train_size = int(0.6 * len(labels))
    test_size = int(0.3 * len(labels))
    print(f"Train size: {train_size}, test size: {test_size}, validation size: {len(labels) - train_size - test_size}")

    train_dataset = CustomDataset(
        texts[:train_size], labels[:train_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = CustomDataset(
        texts[train_size:train_size+test_size], labels[train_size:train_size + test_size]
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=True)

    validation_dataset = CustomDataset(
        texts[train_size+test_size:], labels[train_size + test_size:]
    )
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)
    return train_loader, test_loader, validation_loader



def main(validation_save_path, last_epoch_save_path):
    regexp = re.compile(r'(\b\w*\d\w*\b|[^a-zA-Zа-яА-Я0-9\s])')
    texts, labels = make_dataset_content(regexp, shuffle=True)
    train_loader, test_loader, validation_loader = make_dataloaders(texts, labels)
    neuro_method, classifier_model = auto_create_neuro(
        window_size=256, window_stride=128, num_classes=2,
        path_to_load_model=None, return_classifier=True
    )
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=5e-4)
    neuro_method.train_mod()
    neuro_method.train(
        optimizer, loss, 1, train_loader, validation_loader, validation_save_path, last_epoch_save_path
    ) 
    neuro_method.eval_mod()
    neuro_method.test(test_loader)


if __name__ == "__main__":
    import random
    import json
    import re
    # TODO replace filenames
    main("/home/vodohleb/PycharmProjects/huyna/chep_best.pt", "/home/vodohleb/PycharmProjects/huyna/chep.pt")
    
    

