import pickle
import clip
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random

class clip_classification(nn.Module):
    def __init__(self, clip_model = "ViT-B/32", device = "cuda", 
                triplet_features_file = "./data/caption/annotations/triplets_features.pth",
                category_file = "./data/caption/annotations/triplets_category.txt"):
        super(clip_classification, self).__init__()
        self.clip_model,_ = clip.load(clip_model, device=device)
        self.device = device
        self.logit_scale = self.clip_model.logit_scale
        self.triplet_features = torch.load(triplet_features_file, map_location=device)
        self.neg_dict = self.get_neg_category(category_file)

    def get_neg_category(self, category_file, cluser_num=100):
        with open(category_file,'r') as f:
            cluster_result = f.read().splitlines()
        cluster_dict = {}
        for i in range(cluser_num):
            cluster_dict[i] = []
        for i,cluster in enumerate(cluster_result):
            cluster_dict[int(cluster)].append(i)
        neg_dict = {}
        for i in range(cluser_num):
            neg_dict[i] = []
            for k,v in cluster_dict.items():
                 if k !=i:
                    neg_dict[i].extend(v)
        return neg_dict

    def clip_text(self, caption_gts):
        captions_gt_total = []
        for caption_gt in caption_gts['caption']:
            captions_gt_total.extend(caption_gt)      
        with torch.no_grad():
            text_input = clip.tokenize([f"a photo of {c}" for c in captions_gt_total]).to(self.device)
            text_features = self.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.float()
    
    def loss_batch_contrative_learning(self, outputs_caption_embed, text_features, caption_gts):
        logit_scale = self.logit_scale.exp()
        bs = len(caption_gts['caption'])
        caption_len = [len(c) for c in caption_gts['caption']]
        index = 0
        q_embeds = []
        text_embeds = []
        caption_embed_layers = []
        for i in range(bs):
            caption_embed_layer = outputs_caption_embed[i] / outputs_caption_embed[i].norm(dim=1, keepdim=True)
            caption_embed_layers.append(caption_embed_layer)
            if caption_len[i] == 0:
                bs -= 1
                q_embeds.append([])
                text_embeds.append([])
                continue
            text_feature = text_features[index : index + caption_len[i]]
            index += caption_len[i] 
            with torch.no_grad():
                logits_per_text = text_feature @ caption_embed_layer.t()
                probs = logits_per_text.softmax(dim=-1)
                caption_indice = probs.argmax(dim=-1)
            q_embeds.append(caption_embed_layer[caption_indice,:])
            text_embeds.append(text_feature)

        # query_loss
        loss = 0
        loss_i = 0
        loss_t = 0
        pos = 0
        neg = 0

        for i in range(len(q_embeds)):
            if len(q_embeds[i]) == 0:
                continue
            # i2t loss
            n_t = torch.rand([0,999,512]).to(self.device)
            for index in caption_gts['cluster_category'][i]:
                neg_sample = random.sample(self.neg_dict[index], 999)
                neg_feature = self.triplet_features[neg_sample].unsqueeze(0) 
                n_t = torch.cat([n_t,neg_feature],dim=0)
            text_feature0 = torch.cat((text_embeds[i].unsqueeze(1) , n_t ), dim=1).permute(0,2,1)
            img_feature0 = q_embeds[i].unsqueeze(1)
            logits_per_q = logit_scale * torch.bmm(img_feature0, text_feature0).squeeze(1)
            q_gt = torch.zeros_like(logits_per_q)
            q_gt[:,0] = 1
            loss_i += F.cross_entropy(logits_per_q, q_gt)
            preds = logits_per_q.argmax(dim=-1)
            gts = torch.zeros(logits_per_q.shape[0],dtype=torch.int64)
            for pred,gt in zip(preds,gts):
                if pred == gt:
                    pos += 1
                else:
                    neg += 1

            # t2i loss
            n_q = torch.cat([q_embed for i0,q_embed in enumerate(caption_embed_layers) if i0 != i],dim=0).unsqueeze(0)
            img_feature1 = torch.cat(( q_embeds[i].unsqueeze(1) , n_q.repeat(q_embeds[i].shape[0],1,1) ), dim=1).permute(0,2,1)
            text_feature1 = text_embeds[i].unsqueeze(1)
            logits_per_t = logit_scale * torch.bmm(text_feature1, img_feature1).squeeze(1)
            text_gt = torch.zeros_like(logits_per_t)
            text_gt[:,0] = 1
            loss_t += F.cross_entropy(logits_per_t, text_gt)
        if bs != 0:
            loss_i /= bs
            loss_t /= bs
            loss = (loss_i + loss_t) / 2
            caption_class_error = torch.tensor(1 - (pos / (pos + neg)) , dtype=torch.float32, device=self.device, requires_grad=True)
            return loss, caption_class_error
        else:
            return torch.tensor(loss, dtype=torch.float32, device=self.device, requires_grad=True), torch.tensor(loss, dtype=torch.float32, device=self.device, requires_grad=True)

    def forward(self, outputs_caption_embed, caption_gts):
        caption_features = self.clip_text(caption_gts)
        loss, caption_class_error = self.loss_batch_contrative_learning(outputs_caption_embed, caption_features, caption_gts)
        return loss, caption_class_error

def build_clip_caption_loss(args):
    return clip_classification(args.clip_model, args.device, args.triplet_features_file, args.category_file)

