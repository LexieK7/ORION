import torch
from models.aggregator import BaseAggregator
import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        """
        Fully connected layer module.

        Args:
            in_size (int): Input size.
            out_size (int): Output size. Defaults to 1.
        """
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        """
        Forward pass of the fully connected layer.

        Args:
            feats (torch.Tensor): Input features.

        Returns:
            feats (torch.Tensor): Input features.
            x (torch.Tensor) : Output of the fully connected layer.
        """
        x = self.fc(feats)
        return feats, x
        
        

class MeanPooling(BaseAggregator):
    def __init__(self):
        super(BaseAggregator,self).__init__()


        self.mil=FCLayer(512,2)


    def forward_mil(self,feats):
        #second step: MIL
        
    
        '''
        p_y=self.mil(feats)[1]

        #if p_y.shape[1] ==0:
        #    p_y = torch.tensor([[[0.2, 0.8]]], dtype=torch.float32, device=torch.device('cuda:0'))
        #    print(p_y)
        results = torch.max(p_y,dim=0,keepdim=True)      
        results = results.values
        '''



        #results = torch.unsqueeze(results, dim=0)

        p_y=self.mil(feats)[1]
        #print(p_y.shape)
        results = torch.mean(p_y,dim=0)
        #print(results.shape)
        results = torch.unsqueeze(results, dim=0)
        #print(results.shape)


   
  
        
        return results

    def forward(self, x: torch.Tensor,edge_index2: torch.Tensor=None,edge_index3: torch.Tensor=None):
        """forward model

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjecency matrix
            levels (torch.Tensor): scale level
            childof (torch.Tensor): interscale information
            edge_index2 (torch.Tensor, optional): adjecency matrix of single scale
            edge_index3 (torch.Tensor, optional): adjecency matrix of single scale

        Returns:
            _type_: _description_
        """
        results= self.forward_mil(x)
        return results