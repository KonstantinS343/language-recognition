import torch
from src.summarization.sentence_extraction.embedding_model import EmbeddingModel
from sklearn_extra.cluster import KMedoids



def define_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class SentenceExtractionSummarization:

    def __init__(self, embedding_model: EmbeddingModel):
        self._embedding_model = embedding_model


    @staticmethod
    def _normalize_embeddings(embeddings):
        embeddings_magnitudes = torch.sqrt(
            torch.max(
                torch.sum(torch.square(embeddings), dim=1),
                torch.as_tensor(1e-20)
            )
        )
        for i in range(embeddings.shape[0]):
            embeddings[i] = embeddings[i] / embeddings_magnitudes[i]
        return embeddings


    def summarization(self, sentences, sentences_in_summary):
        sentences_embeddings = self._embedding_model(sentences)
        sentences_embeddings = self._normalize_embeddings(sentences_embeddings)

        clusterizator = KMedoids(n_clusters=sentences_in_summary, init='k-medoids++', metric='euclidean')
        clusterizator.fit(sentences_embeddings.detach().cpu().numpy())

        medoid_indices = clusterizator.medoid_indices_.tolist()
        medoid_indices.sort()
        summarization_sentences = []
        for sentence_index in medoid_indices:
            summarization_sentences.append(sentences[sentence_index])
        return summarization_sentences


    def handle_summarization_request(self, text):
        # TODO split text on sentences
        sentences = [
            "Я не токарь, я не пекарь, я не повар, не доцент",
            "Я не дворник, я не слесарь, я простой советский мент",
            "Я работаю в шараге под названием 'трезвяк'",
            "Вы пашИте не заводах, ну а мне и здесь ништяк",
            "Эй, кто за дверью, выходи в сортир по одному"
        ]
        # TODO remove very short sentences
        if len(sentences) < 5:
            return ". ".join(sentences) + "."
        elif len(sentences) <= 25:
            sentences_in_summary = 5
        else:  # len(sentences) > 25:
            sentences_in_summary = int(20 * len(sentences) / 100)  # 20%
        return ". ".join(self.summarization(sentences, sentences_in_summary)) + "."





if __name__ == "__main__":
    from transformers import BertTokenizerFast, BertModel
    device = define_device()
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased").to(device)
    emb_mod = EmbeddingModel(
        tokenizer=tokenizer,
        model=model,
        window_size=7,
        window_stride=4,
        device=device
    )
    ses = SentenceExtractionSummarization(emb_mod)

    # print(
    #     ses.handle_summarization_request("nigga")
    # )

    text = """
My skin is cold
Transfusion with somebody
Morose and old
Drop into fruitless dying
It was tempting and bared
The whoring angel rising
Now burning prayers
My silent time of losing
My foes, they can't destroy my body
Colliding slow, like life itself
Long for the blur
We cannot dry much longer
Cement to dirt
Disgusted with my cheapness
My foes they can't destroy my body
Colliding slow, like life itself
My foes, my foes they can't destroy my body
Colliding slow, like life itself
My foes they can't destroy my body
Colliding slow, like life itself
My foes, my foes they can't destroy my body
Colliding slow, like life itself, like life itself
    """
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=32, stride=16,
        return_tensors="pt", return_overflowing_tokens=True, padding="max_length"
    )

    for key in tokenized.keys():
        tokenized[key] = tokenized[key].to(device)
    overflow_mapping = tokenized.pop("overflow_to_sample_mapping").detach().cpu().numpy().tolist()

    out = model(**tokenized).last_hidden_state
    max = torch.max(out)
    min = torch.min(out)
    print(out.shape, torch.mean(out).shape)

    print(f"max:\t{max}\nmin:\t{min}\nmean:\t{torch.mean(out)}")


    out = model(**tokenized).last_hidden_state.mean(dim=1)
    max = torch.max(out)
    min = torch.min(out)
    print(out.shape, torch.mean(out).shape)

    print(f"max:\t{max}\nmin:\t{min}\nmean:\t{torch.mean(out)}")







