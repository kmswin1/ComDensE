import torch
import torch.nn as nn
import torch.nn.functional as F

class DensE(nn.Module):
    def __init__(self, args):
        super(DensE, self).__init__()
        self.args = args
        self.ent_emb = nn.Parameter(torch.randn(self.args.num_ent, self.args.embed_dim))
        self.rel_emb = nn.Parameter(torch.randn(2 * self.args.num_rel, self.args.embed_dim))
        self.inp_dropout = nn.Dropout(self.args.inp_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()
        self.transform = nn.Linear((self.args.width + 1) * self.args.matsize, self.args.embed_dim)
        self.ww = nn.ModuleList(
            [nn.Linear(2 * self.args.embed_dim, self.args.matsize) for _ in range(self.args.width)])
        self.w_r0 = nn.Parameter(torch.randn(2 * self.args.num_rel, 2 * self.args.embed_dim * self.args.matsize))
        self.b0 = nn.Parameter(torch.zeros(2 * self.args.num_rel))
        self.bn0 = torch.nn.BatchNorm1d(self.args.matsize)
        self.bn3 = torch.nn.BatchNorm1d(self.args.embed_dim)
        self.bn4 = torch.nn.BatchNorm1d(self.args.width * self.args.matsize)
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.args.num_ent)))
        self.gate1 = nn.Linear(self.args.width * self.args.matsize, 1)
        self.gate2 = nn.Linear(self.args.matsize, 1)
        self.gate3 = nn.Linear(self.args.width * self.args.matsize, 1)
        self.gate4 = nn.Linear(self.args.matsize, 1)

        self.bceloss = torch.nn.BCELoss()

        for w in self.ww:
            torch.nn.init.xavier_normal_(w.weight)
        torch.nn.init.xavier_normal_(self.ent_emb.data)
        torch.nn.init.xavier_normal_(self.rel_emb.data)
        torch.nn.init.xavier_normal_(self.w_r0.data)
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
        w_r0 = w_r0.view(-1, self.args.matsize, 2 * self.args.embed_dim)
        x1 = torch.bmm(w_r0, x.unsqueeze(2)).squeeze(2)
        x1 += self.b0[r].unsqueeze(1)
        x1 = self.hid_dropout(x1)
        x1 = self.bn0(x1)
        #x1 = self.activation(x1)

        x2 = torch.cat([f(x) for f in self.ww], dim=-1)
        x2 = self.hid_dropout(x2)
        x2 = self.bn4(x2)
        #x2 = self.activation(x2)

        a1 = torch.sigmoid(self.gate1(x2))
        a2 = torch.sigmoid(self.gate2(x1))
        #b1 = torch.sigmoid(self.gate3(x2))
        #b2 = torch.sigmoid(self.gate4(x1))
        x = self.transform(torch.cat([a1*x1, a2*x2], dim=-1))
        x = self.hid_dropout(x)
        x = self.bn3(x)
        x = self.activation(x)

        pred = torch.mm(x, self.ent_emb.permute(1, 0))
        pred += self.bias.expand_as(pred)

        pred = torch.sigmoid(pred)
        return pred

class SharedDensE(nn.Module):
    def __init__(self, args):
        super(SharedDensE, self).__init__()
        self.args = args
        self.ent_emb = nn.Parameter(torch.randn(self.args.num_ent, self.args.embed_dim))
        self.rel_emb = nn.Parameter(torch.randn(2*self.args.num_rel, self.args.embed_dim))
        self.inp_dropout = nn.Dropout(self.args.inp_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()
        self.w_r = nn.ModuleList([nn.Linear(2*self.args.embed_dim, 2*self.args.embed_dim) for _ in range(self.args.width)])
        self.transform = nn.Linear(self.args.width*2*self.args.embed_dim, self.args.embed_dim)
        self.bn0 = torch.nn.BatchNorm1d(self.args.width*2*self.args.embed_dim)
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

        if self.args.depth >= 3:
            x = self.w_rr(x)
            x = self.hid_dropout(x)
            x = self.bn1(x)
            x = self.activation(x)

        if self.args.depth >= 4:
            x = self.w_rrr(x)
            x = self.hid_dropout(x)
            x = self.bn2(x)
            x = self.activation(x)

        if self.args.depth >= 5:
            x = self.w_rrr(x)
            x = self.hid_dropout(x)
            x = self.bn3(x)
            x = self.activation(x)

        if self.args.depth >= 6:
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
