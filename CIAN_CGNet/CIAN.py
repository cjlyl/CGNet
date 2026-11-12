import zlib
from support_augmentation import augment_support



import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from AttentionModule import Attention_Module
from Aggerator import *


class CIAN(nn.Module):
    def __init__(self, args, num_symbols, embedding_size, embed, use_pretrain, finetune):
        super(CIAN, self).__init__()

        self.args = args
        self.entity_encoder = Attention_Module(self.args, embed=embed, num_symbols = num_symbols,
                                                embedding_size = embedding_size,
                                                use_pretrain=use_pretrain, finetune=finetune)


        self.Aggerator = SoftSelectPrototype(self.args)


    def encode_support(self, support, support_meta):
        method = getattr(self.args, "support_aug", "none")
        if method == "none":
            return self.entity_encoder(support, support_meta)
        features = self.entity_encoder(support, support_meta, return_entities=True)
        pair_ids = support.detach().cpu().numpy().astype("<i8", copy=False)
        seed = zlib.crc32(pair_ids.tobytes(), self.args.seed & 0xffffffff)
        expanded = augment_support(features.detach().cpu().numpy()[None], method,
                                   self.args.aug_per_support, self.args.aug_sigma,
                                   np.random.default_rng(seed))[0]
        pairs = torch.from_numpy(expanded).to(features.device).flatten(1)
        return self.entity_encoder.bi_MLP(pairs)

    def score_queries(self, support_rep, query, query_meta):
        query_rep = self.entity_encoder(query, query_meta)
        prototype = self.Aggerator(support_rep, query_rep)
        return torch.sum(query_rep * prototype, dim=1)

    def forward(self, support, support_meta,
                query, query_meta,
                false = None, false_meta=None, is_train=True):


        if is_train:

            support_rep = self.entity_encoder(support, support_meta)
            query_rep = self.entity_encoder(query, query_meta)
            false_rep = self.entity_encoder(false, false_meta)
            support_query = self.Aggerator(support_rep, query_rep) #(1, emb_dim) (128,100)
            support_false = self.Aggerator(support_rep, false_rep) #(1, emb_dim)



            positive_score = torch.sum(query_rep * support_query, dim=1) #size:([batch_size])
            negative_score = torch.sum(false_rep * support_false, dim=1)


        else:
            support_rep = self.encode_support(support, support_meta)
            positive_score = self.score_queries(support_rep, query, query_meta)
            negative_score = None
            # sim = None


        return positive_score, negative_score