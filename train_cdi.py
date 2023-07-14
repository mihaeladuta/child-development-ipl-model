import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models
import time
import utils
import models
import training
import io_utils
import train_utils


def batch_update(mapping, nvwd, trlfname, distractors, ptimeincs, 
                 vsplits, vtimeincs, segchar, modelinds,
                 epochstart, epochstop, necrit, padding, precision, gpuindex,
                 phn_ihid_dim, phn_ihid_nlayers, phn_ihid_ndirs, hub_nlayers, hub_ndirs, tgt_full):

    print('Preparing batch update routine')

    # define data and processing
    verbose_data  = True
    verbose_model = True

    train_settings = train_utils.get_setup_train_cdi(mapping, nvwd, trlfname, ptimeincs, 
                                                     vsplits, vtimeincs, segchar, 
                                                     phn_ihid_dim, phn_ihid_nlayers, 
                                                     phn_ihid_ndirs, 
                                                     hub_nlayers, hub_ndirs, tgt_full)
    # assign settings to local variables
    # inputs settings
    nwords     = train_settings['nwords']
    maskval    = train_settings['maskval']
    wrdlist    = train_settings['wrdlist']
    sem_dim    = train_settings['sem_dim']
    vis_dim    = train_settings['vis_dim']
    wrd_nslots = train_settings['wrd_nslots']
    phnsteps   = train_settings['phnsteps']
    vissteps   = train_settings['vissteps']
    semsource  = train_settings['semsource']
    semproc    = train_settings['semproc']
    vissource  = train_settings['vissource']
    visproc    = train_settings['visproc']
    # model settings
    modelparams   = train_settings['modelparams']
    modelreps     = train_settings['modelreps']
    modelinps     = train_settings['modelinps']
    modelproc     = train_settings['modelproc']
    modelarch     = train_settings['modelarch']
    lrate         = train_settings['lrate']
    momentum      = train_settings['momentum']
    nesterov      = train_settings['nesterov']
    wgt_decay     = train_settings['wgt_decay']
    bias          = train_settings['bias']
    nvwd          = train_settings['nvwd']

    if epochstart > 0:
        train_continue  = True;
        continue_epochs = epochstart
        nepochs         = epochstop - epochstart
    else:
        train_continue  = False;
        continue_epochs = np.nan
        nepochs         = epochstop


    # set device to cpu for the creation of the variables etc
    device = torch.device('cpu')

    if precision == 'double':
        dtype = torch.float64
    elif precision == 'single':
        dtype = torch.float32
    elif precision == 'half':
        dtype = torch.float16

    tstart = time.time()

    # read the phone encoding
    phonedata, pcodes, \
    pcats, pipas, pdiscs = io_utils.read_pencodings(train_settings['phnfname'],
                                                    train_settings['phnsname'],
                                                    train_settings['phnind'], 
                                                    index_col='disc', 
                                                    verbose=verbose_data)

    # read the word embeddings and visual features
    semdata, sitems, sfeats = io_utils.read_features(train_settings['semfname'], 
                                                     index_col='word', 
                                                     description='Semantic', 
                                                     verbose=verbose_data)
    visdata, vitems, vfeats = io_utils.read_features(train_settings['visfname'], 
                                                     index_col='word', 
                                                     description='Visual', 
                                                     verbose=verbose_data)

    # read the CDI categories
    catdata, wcats = io_utils.read_cdicategories(train_settings['catfname'], 
                                                 verbose=verbose_data)

    # read the words used for training and associated ipa encoding in disc form
    wrddata, words, wipa, wdiscraw, wlens = io_utils.read_lexicon(train_settings['wrdfname'], 
                                                                  phonedata, catdata, 
                                                                  verbose=verbose_data)

    if not nvwd is None:
        trldata = io_utils.read_traintrls(train_settings)
    else:
        trldata = None

    if segchar != '':
        wdisc = [disc+segchar for disc in wdiscraw]
    else:
        wdisc = wdiscraw

    phn_dim    = pcodes.shape[1]
    wrd_dim    = wrd_nslots*phn_dim

    print()
    print('Representation files')
    print('--------------------')
    print(f'- word list        {wrdlist}')
    print(f'- number of words  {str(len(words))}')
    print(f"- phoneme encoding {train_settings['phnfname']}"\
          f"sheet {train_settings['phnsname']}")
    print(f"- word embeddings  {train_settings['semfname']}")
    print(f"- visual features  {train_settings['visfname']}")
    print(f"- word list        {train_settings['wrdfname']}")

    print()
    print('Representations details')
    print('-----------------------')
    print(f'- phone dimension     {str(phn_dim)}")
    print(f'- semantic dimension  {str(sem_dim)}")
    print(f'- visual dimension    {str(vis_dim)}")
    print(f'- phoneme slots       {str(wrd_nslots)}")
    print(f'- max word length     {str(max(wlens))}")
    print(f'- word form dimension {str(wrd_dim)}")

    print()
    if torch.cuda.is_available():
        print("Allocated GPU memory after feature data read: ",
              '{0:010d}'.format(torch.cuda.memory_allocated()))

    print()
    print("Reading the data took {:12.8e} sec".format(time.time() - tstart))

    # training data
    tstart = time.time()
    inp_phn, inp_vis, phn_lens, out_sem, out_vis, out_wrd,\
    out_sem_v, out_vis_v, out_wrd_v = \
          io_utils.prepare_lexicon_input_output(modelinps, modelreps,
                                                words, wdisc, pcodes, pdiscs,
                                                sfeats, sitems,
                                                vfeats, vitems,
                                                wrd_nslots,
                                                dtype, device,
                                                tgt_full,
                                                maskval,
                                                ptimeincs, vsplits, vtimeincs,
                                                phnsteps, vissteps,
                                                padding, False)

    if torch.cuda.is_available():
        print("Allocated GPU memory after preparing input-output: ", 
              '{0:010d}'.format(torch.cuda.memory_allocated()))


    set_phn, set_vis, set_plens, tgt_sem, tgt_vis, tgt_wrd, \
    tgt_sem_v, tgt_vis_v, tgt_wrd_v, vwd_sem_v, vwd_vis_v, vwd_wrd_v =
          io_utils.prepare_train_sets_tgts(modelinps, modelreps, 
                                           words, trldata, inp_phn, inp_vis, phn_lens, 
                                           out_sem, out_vis, out_wrd,
                                           out_sem_v, out_vis_v, out_wrd_v, 
                                           padding, dtype, device)
    if torch.cuda.is_available():
        print("Allocated GPU memory after preparing training set: ", 
              '{0:010d}'.format(torch.cuda.memory_allocated()))

    del inp_phn
    del inp_vis
    del out_sem
    del out_vis
    del out_wrd
    del out_sem_v
    del out_vis_v
    del out_wrd_v

    if torch.cuda.is_available():
        print("Allocated GPU memory after preparing deleting input-output sets: ", 
              '{0:010d}'.format(torch.cuda.memory_allocated()))


    tgt2sem_maxdist    = np.sqrt(sem_dim)
    tgt2vis_maxdist    = np.sqrt(vis_dim)
    tgt2wrd_maxdist    = np.sqrt(wrd_dim)

    if verbose_data:
        print()
        print('Training data')
        print('-------------')
        print('Inputs')
        utils.describe(set_phn, '- set_phn')
        utils.describe_tensor(set_phn[0], '- set_phn tensors')
        print('Targets')
        if "sem" in modelreps:
            utils.describe(tgt_sem, '- tgt_sem')
            utils.describe_tensor(tgt_sem[0], '- tgt_sem tensors')
        if "vis" in modelreps:
            utils.describe(tgt_vis, '- tgt_vis')
            utils.describe_tensor(tgt_vis[0], '- tgt_vis tensors')
    else:
        print()
        print('Training data defined')

    print()
    print("Preparing the data took {:12.8e} sec".format(time.time() - tstart))

    for imdl in modelinds:

        modelfname = f'model_{modelarch}{modelparams}_{wrdlist}_{str(imdl)}'
        if not trlfname is None:
            outfolder  = f'models/{wrdlist}/{trlfname}/{modelarch}{modelparams}'\
                         f'/SEM_{semsource}_{semproc}_'\
                         f'VIS_{vissource}_{visproc}_P{padding}_{precision}'\
                         f'/model_{str(imdl)}/'
        else:
            outfolder  = f"models/{wrdlist}/{train_settings['lexicon']}"\
                         f"/{modelarch}{modelparams}"\
                         f"/SEM_{semsource}_{semproc}"\
                         f"_VIS_{vissource}_{visproc}_P{padding}_{precision}"\
                         f"/model_{str(imdl)}/"

        # create folders
        if not os.path.isdir(outfolder):
          os.makedirs(outfolder)

        if verbose_model:
            print()
            print('Model')
            print('-----')
            print(f'- architecture: {modelarch}')
            print(f"- model parameters: {modelparams.replace('_', ' ')}")
            print(f'- learning rate: '{lrate}, momentum: {momentum}")
            print(f'- max epochs: {nepochs},  error checkpoint: every {necrit} epochs')
            print(f'- output folder: {outfolder}')
            print(f'- output file: {modelfname}')
        else:
            print()
            print('** Model and training defined **')


        errors_read  = []

        print()
        print(f'Creating model {modelarch} ... ')
        model = models.create_model(modelarch, phn_dim, sem_dim, 
                                    vis_dim, wrd_dim, nvwd,
                                    phn_ihid_dim, phn_ihid_nlayers, phn_ihid_ndirs,
                                    hub_nlayers, hub_ndirs,
                                    bias, verbose=False)
        print('done')

        if torch.cuda.is_available():
            device = torch.device(f'cuda: {gpuindex}')
            print("Initial allocated GPU memory: ",
                  '{0:010d}'.format(torch.cuda.memory_allocated()))
        else:
            device = torch.device('cpu')



        print("Set device to: ", device)
        print("Set dtype to:  ", dtype)

        model = model.to(device=device, dtype=dtype)
        model.train()

        if torch.cuda.is_available():
            print("Allocated GPU memory after sending model to GPU: ", 
                  '{0:010d}'.format(torch.cuda.memory_allocated()))

        set_phn = set_phn.to(device)
        if "vis" in modelinps or "vwd" in modelinps:
            set_vis = set_vis.to(device=device, dtype=dtype)
        if "sem" in modelreps:
            tgt_sem = tgt_sem.to(device=device, dtype=dtype)
        if "vis" in modelreps:
            tgt_vis = tgt_vis.to(device=device, dtype=dtype)
        if "wrd" in modelreps:
            tgt_wrd = tgt_wrd.to(device=device, dtype=dtype)

        # create data structures to pass to training function
        sets = {}
        sets['set_phn']         = set_phn
        sets['set_vis']         = set_vis
        sets['set_plens']       = set_plens
        sets['tgt_sem']         = tgt_sem
        sets['tgt_vis']         = tgt_vis
        sets['tgt_wrd']         = tgt_wrd

        params = {}
        params['mapping']         = mapping
        params['modelreps']       = modelreps
        params['modelinps']       = modelinps
        params['modelproc']       = modelproc
        params['modelarch']       = modelarch
        params['modelfname']      = modelfname
        params['nvwd']            = nvwd
        params['lrate']           = lrate
        params['momentum']        = momentum
        params['wgt_decay']       = wgt_decay
        params['nesterov']        = nesterov
        params['nepochs']         = nepochs
        params['necrit']          = necrit
        params['device']          = device
        params['dtype']           = dtype
        params['padding']         = padding
        params['maskval']         = maskval
        params['train_continue']  = train_continue
        params['continue_epochs'] = continue_epochs
        params['distractors']     = distractors
        params['outfolder']       = outfolder

        training.batch_update(model, sets, params)
