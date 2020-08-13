import numpy as np
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel

def get_many_embeddings(tensors, inference_model, batch_size=128, emb_dim=256):
    embeddings = torch.Tensor(len(tensors), emb_dim)
    for i in range(0, len(tensors), batch_size):
        embeddings[i:i + batch_size] = inference_model.get_embeddings(tensors[i:i + batch_size], None)[0]
    return embeddings

def get_inference_model(trunk, embedder):
    match_finder = MatchFinder(mode="sim", threshold=0.7)
    inference_model = InferenceModel(trunk, embedder=embedder, match_finder=match_finder)
    return inference_model

def get_embeddings(inference_model, dataset):
    return get_many_embeddings([dataset[i][0] for i in range(len(dataset))], inference_model)

def get_scores(inference_model, gallery_embeddings, query_embeddings, gallery_labels, query_labels):
    calculator = AccuracyCalculator()
    scores_dict = calculator.get_accuracy(query_embeddings.numpy(),
                                       gallery_embeddings.numpy(),
                                       query_labels,
                                       gallery_labels,
                                       embeddings_come_from_same_source=False)

    return scores_dict

