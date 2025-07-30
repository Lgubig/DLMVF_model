import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).mean(1), beta


class EnhancedDNN(nn.Module):
    def __init__(self, input_dim,  output_dim, dropout=0.2):
        super(EnhancedDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x= self.dropout(x)
        x = self.fc2(x)
        return x

class GCNNetmuti(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_smile=66,
                 num_features_xt=25, output_dim=128, dropout=0.2):
        super(GCNNetmuti, self).__init__()
        self.enhanced_dnn = EnhancedDNN(512,  output_dim)
        #药物 SMILES character CNN processing
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=4, padding=2)
        self.conv_xd_12 = nn.Conv1d(n_filters* 2, out_channels=n_filters * 4, kernel_size=4, padding=2)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=3, padding=2)
        self.conv_xd_22 = nn.Conv1d(n_filters* 2, out_channels=n_filters * 4, kernel_size=3, padding=2)
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=2, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters* 2, out_channels=n_filters * 4, kernel_size=2, padding=1)
        self.fc_smiles = torch.nn.Linear(n_filters *4, output_dim)

        # 药物SMILES graph branch
        self.n_output = n_output
        self.gcnv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.gcnv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 512)
        self.fc_g2 = torch.nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #
        # Convolution layers for reducing dimensions after concatenation
        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim * 3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=384, out_channels=output_dim, kernel_size=1)


        # miRNA的序列多尺度1DCNN特征提取
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters* 2, out_channels=n_filters * 4, kernel_size=4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=3, padding=2)
        self.conv_xt_22 = nn.Conv1d(n_filters* 2, out_channels=n_filters * 4, kernel_size=3, padding=2)
        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters* 2, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters* 2,out_channels=n_filters * 4, kernel_size=2, padding=1)

        # miRNA与药物关联矩阵数据处理
        self.n_output = n_output
        self.gcn_md1 = GCNConv(256, 128)
        self.gcn_md2 = GCNConv(128, 64)

        self.fc_md1 = torch.nn.Linear(64, 128)
        self.fc_md2 = torch.nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.attention = Attention(output_dim)

        # Combined layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()


    def forward(self, data, data_o):
        # # 药物的图处理
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcnv1(x, edge_index)
        x = self.relu(x)
        x = self.gcnv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 药物的序列处理方法
        drugsmile = data.seqdrug

        embedded_smile = self.smile_embed(drugsmile.long())
        embedded_smile = embedded_smile.permute(0, 2, 1)
        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)

        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)

        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)

        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)

        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, kernel_size=2)

        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 = self.dropout(conv_xd3)
        conv_xd3 = F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)

        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)

        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1)
        conv_xd = conv_xd.unsqueeze(1).permute(0, 2, 1)
        conv_xd = self.conv_reduce_smiles(conv_xd)
        conv_xd = conv_xd.squeeze(2)

        # mrina与基因处理过程
        gene = data.gene.float()
        dnn_xg = self.enhanced_dnn(gene)


        # mirna的序列处理方法
        target = data.target
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)

        conv_xt1 = self.conv_xt_11(embedded_xt)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = self.dropout(conv_xt1)
        conv_xt1 = self.conv_xt_12(conv_xt1)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = F.max_pool1d(conv_xt1, conv_xt1.size(2)).squeeze(2)

        conv_xt2 = self.conv_xt_21(embedded_xt)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = self.dropout(conv_xt2)
        conv_xt2 = self.conv_xt_22(conv_xt2)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 = F.max_pool1d(conv_xt2, conv_xt2.size(2)).squeeze(2)

        conv_xt3 = self.conv_xt_31(embedded_xt)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = self.dropout(conv_xt3)
        conv_xt3 = self.conv_xt_32(conv_xt3)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, conv_xt3.size(2)).squeeze(2)

        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1)
        conv_xt = conv_xt.unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt)
        conv_xt = conv_xt.squeeze(2)

        # GCN对mirna-drug图进行处理
        row_indices = data.row_indices
        col_indices = data.col_indices
        data_o_x, data_o_edge_index,data_o_batch= data_o.x, data_o.edge_index,data_o.batch
        dnn_features = self.gcn_md1(data_o_x, data_o_edge_index)
        dnn_features = self.relu(dnn_features)
        dnn_features = self.gcn_md2(dnn_features, data_o_edge_index)
        dnn_features = self.relu(dnn_features)
        dnn_features=self.fc_md1(dnn_features)
        dnn_features = self.dropout(dnn_features)
        dnn_features = self.fc_md2(dnn_features)
        dnn_features = self.dropout(dnn_features)
        rna_features = dnn_features[row_indices, :]
        drug_features = dnn_features[col_indices + 1563, :]

        # 药物的特征融合方法
        #conv_xd为drug的序列特征
        # x为drug的图特征
        # drug_features为GCN对mirna-drug图进行处理
        drugemb = torch.stack([conv_xd,x,drug_features], dim=1)
        drugemb, att = self.attention(drugemb)
        drugemb = self.dropout(drugemb)
        # print('drugemb的得分情况：',att)
        # drugemb = conv_xd * x * drug_features

        # conv_xt为mirna的特征
        #dnn_xg为mirna与基因的关联
        #rna_feature为mirna-drug图进行处理|
        #conv_xt为序列特征
        miRNAemb = torch.stack([dnn_xg,rna_features,conv_xt], dim=1)
        miRNAemb, att = self.attention(miRNAemb)
        miRNAemb = self.dropout(miRNAemb)
        # print('miRNAemb的得分情况：',att)

        # miRNAemb=dnn_xg*rna_features*conv_xt
        xc = torch.cat((drugemb, miRNAemb), dim=1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)

        # mlp

        return out
