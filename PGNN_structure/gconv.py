from Motif_Position.test_pgnn import *
import numpy as np
from utils.config import cfg
import math
alpha = 0.0001


class Gconv(nn.Module):

    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.new_num_inputs = in_features
        self.new_num_outputs = out_features
        self.new_fc = nn.Linear(self.new_num_inputs, self.new_num_outputs)

        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
        
    def forward(self, A, x, A_src_motif_label_list):
        
        dists_max, dists_argmax, dist_graph = caculate(A)
        temp_a = torch.zeros(len(dists_max), dists_max[0].shape[0], dists_max[0].shape[1])

        A = concateTensor(A)
        #A = torch.add(A, count_entropy(A))

        dists_argmax_arr = np.array(dists_argmax)
        dists_argmax_arr_fla = dists_argmax_arr.flatten()

        dist_max_zero = np.zeros((dists_max[0].shape[0],dists_max[0].shape[1]))
        dist_graph_zero =  torch.zeros(len(dist_graph), dist_graph[0].shape[0], dist_graph[0].shape[1]).to(device)
        for i in range(dist_max_zero.shape[0]):
            dist_max_zero[i] = dists_max[0][i].cpu()
        for i in range(len(dist_graph)):
            dist_graph_zero[i] =  dist_graph[i]

        dists_max_ten = torch.from_numpy(dist_max_zero.astype(np.float32)).to(device)
        subset_features = torch.zeros(x.shape[0],dists_max[0].shape[1],x.shape[2]).to(device)

        for i in range(len(x)):
             subset_features[i] = x[i][dists_argmax_arr_fla[i][0],:] 
             
        dist_sum = torch.zeros(len(x), x.shape[1], 1).to(device)
        for i in range(len(x)):
            for j in range(x.shape[1]):
                for k in range(x.shape[1]):
                    dist_sum[i][j] += math.exp(dist_graph[i][j][k])
    
        
        att = torch.zeros(x.shape[0],1,x.shape[1]).to(device)

        for i in range(len(x)):
            for j in range(x.shape[1]):
                for k in range(x.shape[1]):
                    att[i][0][k] = math.exp(dist_graph[i][j][k])/math.exp(dist_sum[i][j])
 
        for i in range(len(x)):
            for j in range(x.shape[1]):
                subset_features[i][j] *= att[i][0][j]
        #resume2 =False
        #assert resume2,"break"
        
       # message = torch.matmul(dist_graph_zero,subset_features).to(device)

        #au = torch.cat((alpha*message,x),-1).to(device)
        au = alpha*subset_features+x
        au = nn.AdaptiveAvgPool1d(self.num_outputs)(au)
   
        A = F.normalize(A, p=1, dim=-2)
        
        #self.m_fc = nn.Linear(au.shape[2], self.num_outputs).to(device)

        #au = self.a_fc(au)
        #au = nn.AdaptiveAvgPool1d(au.shape[2])(au)
        #ux = self.u_fc(x)
        ux = nn.AdaptiveAvgPool1d(self.num_outputs)(x)

        x = torch.bmm(A, F.relu(au)) +F.relu(ux)  # has size (bs, N, num_outputs)

        
        x2 = A_src_motif_label_list
        x2 = torch.mean(x2.type(torch.FloatTensor), dim=1, keepdim=True)

        x2 = x2.repeat([1, self.new_num_outputs])

        return x, x2


class Siamese_Gconv(nn.Module):

    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
        '''

    def forward(self, g1, g2, g3, g4):
        emb1, emb3 = self.gconv(*g1, *g3)
        emb2, emb4 = self.gconv(*g2, *g4)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2, emb3, emb4
