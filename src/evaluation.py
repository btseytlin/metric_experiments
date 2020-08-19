from tqdm import tqdm
import numpy as np
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity

def get_model_device(model):
    return next(model.parameters()).device

def get_many_embeddings(dataset, inference_model, batch_size=64, emb_dim=256):
    embeddings = torch.zeros(len(dataset), emb_dim)
    tensors = []
    for i, pair in tqdm(enumerate(dataset.iter()), total=len(dataset)):
        tensors.append(pair[0])

        if len(tensors) >= batch_size:
            embeddings[i+1-batch_size:i+1] = inference_model.get_embeddings(torch.stack(tensors).to(get_model_device(inference_model.trunk)), None)[0]
            tensors = []
    
    assert not embeddings.isnan().any().item()
    return embeddings

def get_inference_model(trunk, embedder):
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
    inference_model = InferenceModel(trunk, embedder=embedder, match_finder=match_finder)
    return inference_model

def get_embeddings(inference_model, dataset):
    return get_many_embeddings(dataset, inference_model)

def get_scores(inference_model, gallery_embeddings, query_embeddings, gallery_labels, query_labels):
    calculator = AccuracyCalculator()
    scores_dict = calculator.get_accuracy(query_embeddings.numpy(),
                                       gallery_embeddings.numpy(),
                                       query_labels,
                                       gallery_labels,
                                       embeddings_come_from_same_source=False)

    return scores_dict

