import sys
import train_cdi


def train_wrapper(nvwd, trlfname, distractors, 
                  mapping, ptimeincs, vsplits, vtimeincs, segchar,
                  modelstart, modelstop, 
                  epochstart, epochstop, necrit, 
                  padding, precision, gpuindex,
                  phn_ihid_dim, phn_ihid_nlayers, phn_ihid_ndirs, 
                  hub_nlayers, hub_ndirs, tgt_full):

    modelinds    = list(range(int(modelstart), int(modelstop)+1))

    train_cdi.batch_update(mapping, int(nvwd), trlfname, distractors, 
                           int(ptimeincs), int(vsplits), int(vtimeincs), segchar,
                           modelinds, int(epochstart), int(epochstop), int(necrit), 
                           padding, precision, gpuindex,
                           int(phn_ihid_dim), int(phn_ihid_nlayers), 
                           int(phn_ihid_ndirs), int(hub_nlayers), int(hub_ndirs), 
                           int(tgt_full))


if __name__ == '__main__':
    print('Running   :'+sys.argv[0])
    train_wrapper(*sys.argv[1:])
