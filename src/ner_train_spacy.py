import random

import spacy
from spacy.training import Example


TRAIN_DATA = [
    (
        "Ahmet Yilmaz, Bogazici Universitesi Bilgisayar Muhendisligi mezunu.",
        {"entities": [(0, 12, "NAME"), (14, 59, "EDUCATION")]},
    ),
    (
        "Python, SQL, Streamlit, TensorFlow ile projeler gelistirdim.",
        {"entities": [(0, 6, "SKILL"), (8, 11, "SKILL"), (13, 22, "SKILL"), (24, 34, "SKILL")]},
    ),
]


def train_resume_ner(train_data=None, n_iter=20, model_out="models/spacy_resume_ner"):
    if train_data is None:
        train_data = TRAIN_DATA

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    labels = set()
    for _, ann in train_data:
        for start, end, label in ann.get("entities", []):
            labels.add(label)
    for label in labels:
        ner.add_label(label)

    optimizer = nlp.begin_training()
    for epoch in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, ann in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, ann)
            nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
        print("Epoch {0}/{1} Losses: {2}".format(epoch + 1, n_iter, losses))

    nlp.to_disk(model_out)
    return nlp


if __name__ == "__main__":
    train_resume_ner()
