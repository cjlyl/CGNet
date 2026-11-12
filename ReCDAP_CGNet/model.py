import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import ConditionalUnet1D, DDPMScheduler
from relational_path_gnn import RelationalPathGNN

###################################################
# 1) LSTM_attn : relation_learner (from prior work)
###################################################
class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=self.embed_size * 2,
            hidden_size=self.n_hidden,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
        )
        self.out = nn.Linear(self.n_hidden * 2 * self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).to(lstm_output.device)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weight)
        context = context.view(-1, self.n_hidden * 2 * self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = torch.zeros(self.layers * 2, size[0], self.n_hidden, device=inputs.device)
        cell_state = torch.zeros(self.layers * 2, size[0], self.n_hidden, device=inputs.device)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_cell_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)

###################################################
# 2) ScoreCalculator
###################################################
class ScoreCalculator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.head_encoder = nn.Linear(4 * emb_dim, emb_dim)
        self.tail_encoder = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, h, t, r, pos_num, z):
        """
        h, t, r : (B, nq+nn, 1, emb_dim)
        z       : (B, nq+nn, embed_dim)  -> user code에서는 (B, embed_dim)을 unsqueeze
        pos_num : int, query 개수
        return: p_score (B, pos_num), n_score (B, neg_num)
        """
        z_unsq = z.unsqueeze(2)  # (B, nq+nn, 1, embed_dim)
        
        # 머리/꼬리 임베딩을 투영
        h = h + self.head_encoder(z_unsq)
        t = t + self.tail_encoder(z_unsq)
        
        # L2 norm
        score = -torch.norm(h + r - t, p=2, dim=-1)  # (B, nq+nn)
        
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


###################################################
# 3) Attention Pooler
###################################################
class AttentionPooler(nn.Module):
    def __init__(self, embed_dim=100, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )

        # initialize
        nn.init.xavier_normal_(self.query)

    def forward(self, x):
        """
        x: (B, L, D)
        return: (B, D)
        """
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)   # (B, 1, D)
        
        attn_output, attn_weights = self.attn(q, x, x)
        out = self.mlp(attn_output)       # (B, 1, D)
        return out.squeeze(1)            # (B, D)


###################################################
# 4) Relation-based Conditional Diffusion with Attention Pooling
###################################################
class ReCDAP(nn.Module):
    def __init__(self, g, dataset, parameter):
        super().__init__()
        
        self.device = parameter['device']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.few = parameter['few']

        self.r_path_gnn = RelationalPathGNN(
            g, 
            dataset['ent2id'], 
            len(dataset['rel2emb']), 
            parameter
        )

        self.relation_learner =  LSTM_attn(embed_size=self.embed_dim, n_hidden=parameter['lstm_hiddendim'], out_size=self.embed_dim,
                                              layers=parameter['lstm_layers'], dropout=self.dropout_p)

        self.score_calculator = ScoreCalculator(self.embed_dim)

        self.attn_pooler = AttentionPooler(embed_dim=self.embed_dim * 2, num_heads=1)
        
        num_diffusion_iters = parameter['num_diffusion_iters']
        self.noise_pred_net = ConditionalUnet1D(input_dim=self.embed_dim * 2, global_cond_dim=self.embed_dim * 1 + self.embed_dim * self.few * 2 * 2 + self.few * 2) # relation + support + pos/neg label
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
