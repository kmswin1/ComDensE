import torch
import torch.nn as nn
import torch.nn.functional as F

class RelinearE(nn.Module):
    def __init__(self, args):
        super(RelinearE, self).__init__()
        self.args = args
        self.ent_emb = nn.Parameter(torch.randn(self.args.num_ent, self.args.embed_dim))
        self.rel_emb = nn.Parameter(torch.randn(2*self.args.num_rel, self.args.embed_dim))
        self.inp_dropout = nn.Dropout(self.args.inp_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()
        self.transform = nn.Linear(2*self.args.embed_dim, self.args.embed_dim)
        self.w_r0 = nn.Parameter(torch.randn(2*self.args.num_rel, 2*self.args.embed_dim*2*self.args.embed_dim))
        self.w_r1 = nn.Parameter(torch.randn(2*self.args.num_rel, 2*self.args.embed_dim*2*self.args.embed_dim))
        self.w_r2 = nn.Parameter(torch.randn(2*self.args.num_rel, 2*self.args.embed_dim*2*self.args.embed_dim))
        self.b0 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.b1 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.b2 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.bn0 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn1 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn2 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.args.embed_dim)
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.args.num_ent)))

        self.bceloss = torch.nn.BCELoss()

        torch.nn.init.xavier_normal_(self.ent_emb.data)
        torch.nn.init.xavier_normal_(self.rel_emb.data)
        torch.nn.init.xavier_normal_(self.w_r0.data)
        torch.nn.init.xavier_normal_(self.w_r1.data)
        torch.nn.init.xavier_normal_(self.w_r2.data)
        torch.nn.init.xavier_normal_(self.transform.weight)

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, h, r, neg_ents, strategy):
        head = self.ent_emb[h]
        rel = self.rel_emb[r]
        x = self.inp_dropout(torch.cat([head, rel], dim=-1))
        w_r0 = self.w_r0[r]
        w_r1 = self.w_r1[r]
        w_r2 = self.w_r2[r]
        w_r0 = w_r0.view(-1, 2*self.args.embed_dim, 2*self.args.embed_dim)
        w_r1 = w_r1.view(-1, 2*self.args.embed_dim, 2*self.args.embed_dim)
        w_r2 = w_r2.view(-1, 2*self.args.embed_dim, 2*self.args.embed_dim)
        x = torch.bmm(w_r0, x.unsqueeze(2)).squeeze(2)
        x += self.b0[r].unsqueeze(1)
        x = self.hid_dropout(x)
        x = self.bn0(x)
        x = self.activation(x)
        x = torch.bmm(w_r1, x.unsqueeze(2)).squeeze(2)
        x += self.b1[r].unsqueeze(1)
        x = self.hid_dropout(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.transform(x)
        x = self.hid_dropout(x)
        x = self.bn3(x)
        x = self.activation(x)

        pred = torch.mm(x, self.ent_emb.permute(1,0))
        pred += self.bias.expand_as(pred)

        pred = torch.sigmoid(pred)
        return pred

class MultiHead(nn.Module):
    def __init__(self, args):
        super(MultiHead, self).__init__()
        self.args = args
        self.ent_emb = nn.Parameter(torch.randn(self.args.num_ent, self.args.embed_dim))
        self.rel_emb = nn.Parameter(torch.randn(2*self.args.num_rel, self.args.embed_dim))
        self.inp_dropout = nn.Dropout(self.args.inp_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()
        self.w_r = nn.ModuleList([nn.Linear(2*self.args.embed_dim, 2*self.args.embed_dim) for _ in range(self.args.heads)])
        self.transform = nn.Linear(self.args.heads*2*self.args.embed_dim, self.args.embed_dim)
        self.bn0 = torch.nn.BatchNorm1d(self.args.heads*2*self.args.embed_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.args.embed_dim)
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.args.num_ent)))
        self.bceloss = torch.nn.BCELoss()

        torch.nn.init.xavier_normal_(self.ent_emb.data)
        torch.nn.init.xavier_normal_(self.rel_emb.data)
        torch.nn.init.xavier_normal_(self.transform.weight)

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, h, r, neg_ents, strategy):
        head = self.ent_emb[h]
        rel = self.rel_emb[r]
        x = self.inp_dropout(torch.cat([head, rel], dim=-1))
        x = torch.cat([w(x) for w in self.w_r], dim=-1)
        x = self.hid_dropout(x)
        x = self.bn0(x)
        x = self.activation(x)

        x = self.transform(x)
        x = self.hid_dropout(x)
        x = self.bn3(x)
        x = self.activation(x)

        pred = torch.mm(x, self.ent_emb.permute(1,0))
        pred += self.bias.expand_as(pred)

        pred = torch.sigmoid(pred)
        return pred

class MultiLayer(nn.Module):
    def __init__(self, args):
        super(MultiLayer, self).__init__()
        self.args = args
        self.ent_emb = nn.Parameter(torch.randn(self.args.num_ent, self.args.embed_dim))
        self.rel_emb = nn.Parameter(torch.randn(2*self.args.num_rel, self.args.embed_dim))
        self.inp_dropout = nn.Dropout(self.args.inp_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()
        self.w_r = nn.Linear(2*self.args.embed_dim, 2*self.args.embed_dim)
        self.w_rr = nn.Linear(2*self.args.embed_dim, 2*self.args.embed_dim)
        self.w_rrr = nn.Linear(2 * self.args.embed_dim, 2*self.args.embed_dim)
        self.w_rrrr = nn.Linear(2 * self.args.embed_dim, 2*self.args.embed_dim)
        self.w_rrrrr = nn.Linear(2 * self.args.embed_dim, 2*self.args.embed_dim)
        self.transform = nn.Linear(2*self.args.embed_dim, self.args.embed_dim)
        self.b0 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.b1 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.b2 = nn.Parameter(torch.zeros(2*self.args.num_rel))
        self.bn0 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn1 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn2 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn3 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn4 = torch.nn.BatchNorm1d(2*self.args.embed_dim)
        self.bn5 = torch.nn.BatchNorm1d(self.args.embed_dim)
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.args.num_ent)))

        self.bceloss = torch.nn.BCELoss()

        torch.nn.init.xavier_normal_(self.ent_emb.data)
        torch.nn.init.xavier_normal_(self.rel_emb.data)
        torch.nn.init.xavier_normal_(self.transform.weight)

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0]
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, h, r, neg_ents, strategy):
        head = self.ent_emb[h]
        rel = self.rel_emb[r]
        x = self.inp_dropout(torch.cat([head, rel], dim=-1))

        x = self.w_r(x)
        x = self.hid_dropout(x)
        x = self.bn0(x)
        x = self.activation(x)

        if self.args.layers >= 3:
            x = self.w_rr(x)
            x = self.hid_dropout(x)
            x = self.bn1(x)
            x = self.activation(x)

        if self.args.layers >= 4:
            x = self.w_rrr(x)
            x = self.hid_dropout(x)
            x = self.bn2(x)
            x = self.activation(x)

        if self.args.layers >= 5:
            x = self.w_rrr(x)
            x = self.hid_dropout(x)
            x = self.bn3(x)
            x = self.activation(x)

        if self.args.layers >= 6:
            x = self.w_rrr(x)
            x = self.hid_dropout(x)
            x = self.bn4(x)
            x = self.activation(x)

        x = self.transform(x)
        x = self.hid_dropout(x)
        x = self.bn5(x)
        x = self.activation(x)

        pred = torch.mm(x, self.ent_emb.permute(1,0))
        pred += self.bias.expand_as(pred)

        pred = torch.sigmoid(pred)
        return pred
