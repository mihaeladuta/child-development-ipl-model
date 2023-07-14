import torch
import torch.nn as nn
import utils


def create_model(modelarch, phn_dim, sem_dim, vis_dim, wrd_dim, nvwd,
                 phn_ihid_dim, phn_ihid_nlayers, phn_ihid_ndirs,
                 hub_nlayers, hub_ndirs,
                 bias, verbose=False):
    
    model = mapping_phnvwd2sem(phn_dim, sem_dim, vis_dim, nvwd,
                               phn_ihid_dim=phn_ihid_dim, 
                               phn_ihid_nlayers=phn_ihid_nlayers,
                               phn_ihid_ndirs=phn_ihid_ndirs,
                               out_nlayers=hub_nlayers,   
                               out_ndirs=hub_ndirs,
                               bias=bias, verbose=False)


    return model

class mapping_phnvwd2sem(nn.Module):
    """
        Model for an architecture to map from unfolding phonological
        representation and a visual world display to semantic representations
        (PHN, VWD) -> (PHNIHID - optional) -> OUT -> (SEM)
        VWD - visual world display: currently with only 2 items
    """
    def __init__(self, phn_dim, sem_dim, vis_dim, nvwdimgs, recurrent_reps=False,
                 phn_ihid_dim=40, phn_ihid_nlayers=None, phn_ihid_ndirs=None,
                 out_nlayers=1, out_ndirs=1,
                 bias=False, verbose=False):
        """
        Args:
            phn_dim (int)          : dimension for phonological feature vectors
            sem_dim (int)          : dimension for semantic feature vectors
            vis_dim (int)          : dimension for visual features vectors
            nvwdimgs (int)         : number of images in the visual world display
            recurrent_reps (bool)  : recurrency for internal representations
            phn_ihid_dim (int)     : dimension of phonological hidden layer
            phn_ihid_nlayers (int) : number of layers in the hidden gru
            phn_ihid_ndirs (int)   : number of directions in the hidden gru
            out_nlayers (int)      : number of layers in the projection hub
            out_ndirs (int)        : number of direction in the projection hub
            bias (bool)            : whether or not to use bias (default False)
            verbose (bool)         : print information (default False)
        """
        super(mapping_phnvwd2sem, self).__init__()

        # SETTINGS
        # ########s
        # dimensions for features
        self.phn_dim          = phn_dim
        self.sem_dim          = sem_dim
        self.vis_dim          = vis_dim
        self.nvwdimgs         = nvwdimgs
        self.recurrent_reps   = recurrent_reps

        # layer for phonological processing
        self.inp_dim = self.phn_dim + self.nvwdimgs * self.vis_dim
        self.phn_ihid_nlayers = phn_ihid_nlayers
        self.phn_ihid_ndirs   = phn_ihid_ndirs
        if self.phn_ihid_nlayers == 0:
            self.phn_ihid_dim = phn_dim + nvwdimgs * vis_dim
        else:
            self.phn_ihid_dim = phn_ihid_dim
        if phn_ihid_ndirs == 2:
            self.phn_ihid_bi = True
        else:
            self.phn_ihid_bi = False

        # output layer
        self.out_dim          = self.nvwdimgs * self.sem_dim
        self.out_nlayers      = out_nlayers
        self.out_ndirs        = out_ndirs
        if out_ndirs == 2:
            self.out_bi = True
        else:
            self.out_bi = False

        # bias
        self.bias             = bias

        # ARCHITECTURE
        # ############
        # recurrent layer to process unfolding word, one phoneme at a time
        if self.phn_ihid_nlayers > 0:
            self.phn_ihid = nn.GRU(self.inp_dim,
                                   self.phn_ihid_dim,
                                   num_layers=self.phn_ihid_nlayers,
                                   bidirectional=self.phn_ihid_bi,
                                   bias=bias)
        if self.out_nlayers == 0:
            self.out      = nn.Linear(self.phn_ihid_dim, self.out_dim)
        else:
            self.out      = nn.GRU(self.phn_ihid_dim, self.out_dim,
                                   num_layers=self.out_nlayers,
                                   bidirectional=self.out_bi,
                                   bias=bias)
        self.out_nl       = nn.Sigmoid()

        if verbose:
            print("Built model")
            print(self)
            utils.describe_model(self)

    def forward(self, inp_feat=torch.Tensor()):
        """

        Args:
            inp_feat (torch.Tensor): vector with phonological features
        Returns:
            none
        """

        if self.phn_ihid_nlayers > 0:
            phn_ihid_output, phn_ihid_states = self.phn_ihid(inp_feat)
            if self.phn_ihid_bi:
                phn_ihid_output = phn_ihid_output[:, :, self.phn_ihid_dim:]
        else:
            phn_ihid_states = torch.Tensor()
            phn_ihid_output = inp_feat
        if self.out_nlayers == 0:
            output           = self.out(phn_ihid_output)
        else:
            output, states   = self.out(phn_ihid_output)
            if self.out_bi:
                output = output[:, :, self.out_dim:]
        self.output = self.out_nl(output)

        return phn_ihid_output, phn_ihid_states, output
