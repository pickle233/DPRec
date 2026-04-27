import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class DPRec(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(DPRec, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.\
            attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        # parameter for contrastive learning
        self.ssl_temp = 0.2
        self.ssl_reg = 1e-6
        self.ssl_reg_inter = 1e-6
        self._init_weight()
        self.user_p_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
        )

        self.p_proj_layer = nn.Linear(self.dim, 2 * self.dim)
    def forward(
            self,
            items: torch.LongTensor,
            user_triple_set: list,
            item_potential_triple_set: list,
            user_potential_triple_set: list,
            item_origin_triple_set: list,
            user_freq_tensor,
    ):

        user_embeddings = []
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        user_pref_tensor = self.entity_emb(user_triple_set[0][0])

        p_feat = self.user_p_mlp(user_pref_tensor)
        user_p = p_feat.mean(dim=1)  # [B, D]

        if user_freq_tensor is not None:
            freq_weights = user_freq_tensor.unsqueeze(-1)  # [B, T, 1]
            user_p = (p_feat * freq_weights).sum(dim=1) / (freq_weights.sum(dim=1) + 1e-8)
        else:
            user_p = p_feat.mean(dim=1)

        # [batch_size, dim]
        user_intial_embedding = user_emb_0.mean(dim=1)
        user_embeddings.append(user_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb, user_p)
            user_p =user_p+user_emb_i
            user_embeddings.append(user_emb_i)

        item_embeddings = []
        user_p = p_feat.mean(dim=1)
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        # item_embeddings.append(item_emb_origin)
        item_emb_0 = self.entity_emb(item_potential_triple_set[0][0])
        item_intial_embedding = item_emb_origin
        item_embeddings.append(item_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_potential_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_potential_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_potential_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb, user_p)
            user_p = user_p + item_emb_i
            item_embeddings.append(item_emb_i)

        user_potential_embeddings = []
        user_p = p_feat.mean(dim=1)
        user_potential_embeddings_0 = self.entity_emb(user_potential_triple_set[0][0])
        user_intial_potential_embedding = user_potential_embeddings_0.mean(dim=1)
        user_potential_embeddings.append(user_intial_potential_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_potential_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_potential_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_potential_triple_set[2][i])
            # [batch_size, dim]
            user_potential_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb, user_p)
            user_p = user_p + user_potential_emb_i
            user_potential_embeddings.append(user_potential_emb_i)

        item_origin_embeddings = []
        user_p = p_feat.mean(dim=1)
        item_origin_embeddings_0 = self.entity_emb(item_origin_triple_set[0][0])
        item_intial_origin_embedding = item_origin_embeddings_0.mean(dim=1)
        item_origin_embeddings.append(item_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_origin_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_origin_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_origin_triple_set[2][i])
            # [batch_size, dim]
            item_origin_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb, user_p)
            user_p = user_p + item_origin_emb_i
            item_origin_embeddings.append(item_origin_emb_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
            # [batch_size, triple_set_size, dim]
            item_emb_0 = self.entity_emb(item_potential_triple_set[0][0])
            # [batch_size, dim]
            item_embeddings.append(item_emb_0.mean(dim=1))

        user_path_embeddings = self.build_path_embedding(user_triple_set)
        user_potential_path_embeddings = self.build_path_embedding(user_potential_triple_set)

        item_potential_path_embeddings = self.build_path_embedding(item_potential_triple_set)
        item_origin_path_embeddings = self.build_path_embedding(item_origin_triple_set)
        scores = self.predict(
            user_embeddings, item_embeddings,
            user_potential_embeddings, item_origin_embeddings,
            user_path_embeddings, user_potential_path_embeddings,
            item_potential_path_embeddings, item_origin_path_embeddings,
            user_p
        )

        return scores

    def build_path_embedding(self, triple_set):

        path_embeddings = []
        for i in range(self.n_layer):
            h_emb = self.entity_emb(triple_set[0][0])
            path_rel = self.relation_emb(triple_set[1][0])

            for k in range(1, i + 1):
                h_emb += self.entity_emb(triple_set[0][k])
                path_rel *= self.relation_emb(triple_set[1][k])


            path_emb = torch.cat([h_emb.mean(1), path_rel.mean(1)], dim=-1)
            path_embeddings.append(path_emb)

        return path_embeddings

    def predict(
            self,
            user_embeddings, item_embeddings,
            user_potential_embeddings, item_origin_embeddings,
            user_path_embeddings, user_potential_path_embeddings,
            item_potential_path_embeddings, item_origin_path_embeddings,
            user_p
    ):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        e_p_u = user_potential_embeddings[0]
        e_o_v = item_origin_embeddings[0]
        loss = 0
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")

            all_user_emb_pool = torch.cat(user_embeddings, dim=0)

            for i in range(self.n_layer):
                loss += self.path_level_contrastive_loss(
                    user_embeddings[i], user_potential_embeddings[i],
                    user_embeddings, user_potential_embeddings,
                    user_path_embeddings[i], user_potential_path_embeddings[i],
                    user_path_embeddings, user_potential_path_embeddings
                )

                loss += self.path_level_contrastive_loss(
                    item_embeddings[i], item_origin_embeddings[i],
                    item_embeddings, item_origin_embeddings,
                    item_potential_path_embeddings[i], item_origin_path_embeddings[i],
                    item_potential_path_embeddings, item_origin_path_embeddings
                )
                p_loss = self.preference_guided_contrastive_loss(
                    user_embeddings[i],
                    user_p,
                    all_user_emb_pool,
                    layer_idx=i
                )
                loss += self.ssl_reg * p_loss
            anchor_emb = user_embeddings[0]



            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
            for i in range(1, len(user_potential_embeddings)):
                e_p_u = torch.cat((user_potential_embeddings[i], e_p_u), dim=-1)
            for i in range(1, len(item_origin_embeddings)):
                e_o_v = torch.cat((item_origin_embeddings[i], e_o_v), dim=-1)

            e_u = torch.cat((e_u, e_p_u), dim=-1)
            e_v = torch.cat((e_o_v, e_v), dim=-1)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores, loss

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg

    def _init_weight(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _knowledge_attention(self, h_emb, r_emb, t_emb, user_p):

        # [B, T, D]
        user_p = user_p.unsqueeze(1)  # [B, 1, D]
        h_r = torch.cat([h_emb, r_emb], dim=-1)  # [B, T, 2D]


        p_proj = self.p_proj_layer(user_p)  # [B, 1, 2D]
        cross_input = h_r * p_proj  # [B, T, 2D]


        att_weights = self.attention(cross_input).squeeze(-1)  # [B, T]
        att_weights_norm = F.softmax(att_weights, dim=-1)

        # emb_i = (att_weights_norm.unsqueeze(-1) * t_emb).sum(dim=1)  # [B, D]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb).sum(dim=1)

        return emb_i  # [batch_size, dim]



    def path_level_contrastive_loss(
            self,
            current_user_embeddings, current_item_embeddings,
            user_embedding, user_embedding_h,
            user_path_emb=None, user_potential_path_emb=None,
            all_user_path_embs=None, all_user_potential_path_embs=None
    ):

        if user_path_emb is not None and user_potential_path_emb is not None:
            u = F.normalize(user_path_emb, dim=1)
            v = F.normalize(user_potential_path_emb, dim=1)
            all_u = F.normalize(torch.cat(all_user_path_embs, dim=0), dim=1)
            all_v = F.normalize(torch.cat(all_user_potential_path_embs, dim=0), dim=1)

            sim_pos_u = (u * v).sum(dim=1)
            sim_all_u = torch.matmul(u, all_u.T)
            pos_score_u = torch.exp(sim_pos_u / self.ssl_temp)
            ttl_score_u = torch.exp(sim_all_u / self.ssl_temp).sum(dim=1)
            loss_u = -torch.log(pos_score_u / ttl_score_u).sum()

            sim_pos_v = (v * u).sum(dim=1)
            sim_all_v = torch.matmul(v, all_v.T)
            pos_score_v = torch.exp(sim_pos_v / self.ssl_temp)
            ttl_score_v = torch.exp(sim_all_v / self.ssl_temp).sum(dim=1)
            loss_v = -torch.log(pos_score_v / ttl_score_v).sum()

            path_loss = self.ssl_reg_inter * (loss_u + loss_v)

            ssl_loss = path_loss

        return ssl_loss


    def preference_guided_contrastive_loss(self, anchor_emb, p, candidate_embs, layer_idx):

        anchor = F.normalize(anchor_emb, dim=-1)  # [B, D]
        p = F.normalize(p, dim=-1)  # [B, D]
        candidate = F.normalize(candidate_embs, dim=-1)  # [N, D]

        B = anchor.size(0)
        N = candidate.size(0)


        start = layer_idx * B
        end = start + B


        mask_self = torch.zeros((B, N), dtype=torch.bool, device=anchor.device)
        row_idx = torch.arange(B, device=anchor.device)
        col_idx = start + row_idx
        mask_self[row_idx, col_idx] = True


        sim_p = torch.matmul(p, candidate.T)  # [B, N]
        sim_p = sim_p.masked_fill(mask_self, float('-inf'))
        weights = F.softmax(sim_p / self.ssl_temp, dim=1)  # [B, N]`
        pos_emb = torch.matmul(weights, candidate)  # [B, D]
        pos_emb = F.normalize(pos_emb, dim=-1)


        pos_logits = (anchor * pos_emb).sum(dim=-1) / self.ssl_temp  # [B]
        pos_score = torch.exp(pos_logits)


        all_logits = torch.matmul(anchor, candidate.T) / self.ssl_temp  # [B, N]
        all_logits = all_logits.masked_fill(mask_self, float('-inf'))
        ttl_score = torch.exp(all_logits).sum(dim=1)  # [B]

        loss = -torch.log(pos_score / (ttl_score + 1e-8)).mean()
        return loss


