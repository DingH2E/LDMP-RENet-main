import torch
import torch.nn as nn

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self,num_state, num_node, bias = False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node,num_node,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =nn.Conv1d(num_state,num_state,kernel_size=1,bias=bias)
        self.conv3 = nn.Conv1d(num_node, int(num_node/2), kernel_size=1)


    def forward(self, x):
        '''input:(n, num_state, num_node)'''
        h =self.conv1(x.permute(0,2,1).contiguous()).permute(0,2,1)
        h =h+x
        h = self.conv2(self.relu(h))
        h = self.conv3(h.permute(0,2,1).contiguous()).permute(0,2,1)
        return h




class MPR(nn.Module):
    """
    Multi-Prototype Reasoning
    """
    def __init__(self, num_in, num_mid, # num_in=256 num_mid=128
                 Conv2d=nn.Conv2d,
                 BatchNorm2d=nn.BatchNorm2d,
                 normalize=False):
        super(MPR, self).__init__()
        self.normalize = normalize
        self.num_s =int(2*num_mid)
        self.num_n =int(2*num_mid)
        self.num_in = num_in
        # reduce dim
        self.conv_state = Conv2d(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = Conv2d(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(self.num_s, self.num_n*2)    # num_s=256 num_n*2=512
        # -------
        # extend dimension
        self.conv_extend = Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        self.blocker = BatchNorm2d(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x,y):
        '''
        :param x: supp(n, c, h, w)
        :param y: query(n, c ,h, w)
        '''

        n= x.size(0)
        # (n, num_in, h, w) --> (n, d, h, w)
        #                   --> (n, d, l)   l=h*w
        # f_v -> P_{d}
        x_d = (self.conv_state(x)).view(n, self.num_s, -1)    # multi-prototype representation
        y_d = (self.conv_state(y)).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, v, h, w)
        #                   --> (n, v, l)
        # f_d -> P_{v}
        x_v = self.conv_proj(x).view(n, self.num_n, -1)
        y_v = self.conv_proj(y).view(n, self.num_n, -1)
        # Reflect
        y_v_r = y_v

        # projection: feature space -> graph space
        # (n, d, l) x (n, l, v)T --> (n, d, v)
        # (n, v, l) x (n, l, d)T --> (n, v, d)
        g_v = torch.matmul(x_d, y_v.permute(0, 2, 1))
        g_d = torch.matmul(y_d, x_v.permute(0, 2, 1))
        g_sq = torch.cat([g_v, g_d],2)

        # reasoning: (n, d, v) -> (n, d, v)
        g = self.gcn(g_sq)

        # reverse projection: graph space -> feature space
        # (n, d, v) x (n, v, l) --> (n, d, l)
        g = torch.matmul(g, y_v_r)
        # (n, num_state, h*w) --> (n, num_state, h, w)
        g = g.view(n, self.num_s,*x.size()[2:])
        # (n, num_state, h, w) -> (n, num_in, h, w)
        main = y + self.blocker(self.conv_extend(g))
        return main
