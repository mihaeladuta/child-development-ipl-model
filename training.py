import torch
import torch.nn
import os
import time
import numpy as np
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial as spatial


def batch_update(model, sets, params):

    # decode inputs for local use
    set_phn         = sets['set_phn']
    set_vis         = sets['set_vis']
    set_plens       = sets['set_plens']

    tgt_sem         = sets['tgt_sem']
    tgt_wrd         = sets['tgt_wrd']
    tgt_vis         = sets['tgt_vis']

    mapping         = params['mapping']
    modelreps       = params['modelreps']
    modelfname      = params['modelfname']
    nvwd            = params['nvwd']
    lrate           = params['lrate']
    momentum        = params['momentum']
    wgt_decay       = params['wgt_decay']
    nesterov        = params['nesterov']
    nepochs         = params['nepochs']
    necrit          = params['necrit']
    device          = params['device']
    dtype           = params['dtype']
    padding         = params['padding']
    maskval         = params['maskval']
    train_continue  = params['train_continue']
    continue_epochs = params['continue_epochs']
    distractors     = params['distractors']
    outfolder       = params['outfolder']

    # derive dimensions
    ntrls   = set_phn.shape[1]
    vis_dim = set_vis.shape[2]
    if not nvwd is None:
        vis_dim = vis_dim 


    # set the criterion
    criterion = torch.nn.MSELoss(reduction='mean')
    # set the optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lrate, 
                                momentum=momentum,
                                weight_decay=wgt_decay, 
                                nesterov=nesterov)

    print()
    if(train_continue):
        print(f'Continue model training for {nepochs} epochs')
        # load the serialised model object
        infname = f"{outfolder}{modelfname}_E{'{0:07d}'.format(continue_epochs)}.pt"
        state = torch.load(infname)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        # load the errors file
        infname = f"{outfolder}{modelfname}_E{'{0:07d}'.format(continue_epochs)}_training.csv"
        print('Loaded model to continue training ...  '+infname)
        training_read = pd.read_csv(infname, header=None).values
        errors_read    = training_read[:, 0]
    else:
        print(f'Started model training for {nepochs)} epochs')


    epoch_errors          = np.empty(nepochs) * np.nan
    set_vwd = set_vis

    # create the inputs and targets
    print(f"mapping: {mapping}")
    
    inp = torch.cat((set_phn, set_vwd), 2)
    tgt = tgt_sem

    if torch.cuda.is_available():
        print("Allocated GPU memory before sending data and model to GPU: ",
              '{0:010d}'.format(torch.cuda.memory_allocated()))

    # set target and dtype
    inp.to(device=device, dtype=dtype)
    tgt.to(device=device, dtype=dtype)
    if dtype == torch.float16:
        model.half()  # convert to half precision
    else:
        model.to(device=device, dtype=dtype)

    model.train()

    if torch.cuda.is_available():
        print("Allocated GPU memory after sending data and model to GPU: ", 
              '{0:010d}'.format(torch.cuda.memory_allocated()))


    for epoch in range(nepochs):
        tstart = time.time()

        _, _, out = model(inp_feat=inp)
        error = criterion(out, tgt)
        model.zero_grad()
        error.backward()
        optimizer.step()
        epoch_errors[epoch] = error.detach().item()

        if (epoch + 1) % necrit == 0:
            if(train_continue):
                nepochs2print = epoch + continue_epochs + 1
            else:
                nepochs2print = epoch + 1
            if(train_continue):
                errors2print = np.append(errors_read, epoch_errors)
            else:
                errors2print  = epoch_errors

            outfname = modelfname + '_E' + '{0:07d}'.format(nepochs2print)

            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, outfolder + outfname + '.pt')

            print(f'Saved model in {outfolder}{outfname}')

            d4training = errors2print
            np.savetxt(f'{outfolder}{outfname}_training.csv',
                       d4training, delimiter=",")
            # delete file with training data from previous reporting
            if (nepochs2print - necrit) > 0:
                fname = outfolder + modelfname
                fname += f"_E{'{0:07d}'.format(nepochs2print - necrit)}"
                fname += '_training.csv'
                if os.path.exists(fname):
                    os.remove(fname)

