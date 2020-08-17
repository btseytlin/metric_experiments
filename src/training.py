import os
import logging
import matplotlib.pyplot as plt
import umap
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
from cycler import cycler
import torch
from pytorch_metric_learning import losses, miners, samplers, trainers, testers

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

def get_optimizers(trunk, embedder):
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001, weight_decay=0.0001)
    return trunk_optimizer, embedder_optimizer

def get_loss():
    return losses.TripletMarginLoss(margin=0.1)

def get_miner():
    return miners.TripletMarginMiner(margin=0.2, type_of_triplets='all')

def get_sampler(dataset):
    return samplers.MPerClassSampler(dataset.targets, m=4, length_before_new_iter=len(dataset))

def get_testing_hooks(experiment_id, val_dataset, test_interval, patience):
    experiment_dir = os.path.join('experiment_logs', experiment_id)
    record_keeper, _, _ = logging_presets.get_record_keeper(experiment_dir, os.path.join('experiment_logs', 'tensorboard', experiment_id))
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = experiment_dir

    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(20,15))
        plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)   
        plt.show()
        
        writer = SummaryWriter(log_dir=os.path.join('experiment_logs', 'tensorboard', experiment_id))
        writer.add_embedding(umap_embeddings, metadata=labels)
        writer.close()

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, 
                                                visualizer = umap.UMAP(), 
                                                visualizer_hook = visualizer_hook,
                                                dataloader_num_workers = 32)
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, 
                                                dataset_dict, 
                                                model_folder, 
                                                test_interval = test_interval,
                                                patience = patience)
    return end_of_epoch_hook, hooks.end_of_iteration_hook
