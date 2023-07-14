import io_utils


def get_setup_train_cdi(mapping, nvwd, trlfname, ptimeincs, vsplits, vtimeincs, 
                        segchars, phn_ihid_dim, phn_ihid_nlayers, phn_ihid_ndirs, 
                        hub_nlayers, hub_ndirs, tgt_full):

    settings   = {}
    wrdlist    = 'cdi'
    nwords     = 200
    foldername = 'lexicons/'

    # phoneme features
    psettings = io_utils.get_setup_phon_features(f'{foldername}{wrdlist}/')
    settings['phnfname']   = psettings['phnfname']
    settings['phnsname']   = psettings['phnsname']
    settings['phnind']     = psettings['phnind']
    settings['phnsteps']   = psettings['phnsteps']

    # word form
    settings['wrd_nslots'] = 10

    # training parameters
    settings['lrate']           = 0.4
    settings['momentum']        = 0.4
    settings['nesterov']        = True 
    settings['wgt_decay']       = 0
    phn_dim = 20
    wrd_dim = settings['wrd_nslots'] * phn_dim

    digitize     = True
    digitization = 'median'

    # semantic features
    semproc    = 'zscoreclean'
    semsource  = 'glove6B'
    sem_dim    = 100
    # visual features
    visproc    = 'avgpool_F512_zscoreclean_scaled'
    vissource  = 'resnet18'
    vfeat_dim  = 150
    vis_pca    = True
    vissolver  = 'full'
    visrand    = 10
    vissteps   = []
    if vsplits > 1:
        vis_dim = vfeat_dim 
    else:
        vis_dim = vfeat_dim
    vissteps = [1]*vtimeincs

    if vis_pca:
        visproc += '_PCA'
        visproc    = ''.join([visproc, str(vfeat_dim), vissolver])
        if (vissolver == 'arpack') or (vissolver == 'randomized'):
            visproc    = ''.join([visproc, 'R', str(visrand)])

    # build the semantic and visual features file names
    semproc = ''.join(['F', str(sem_dim), '_', semproc])

    if digitize:
        semproc = ''.join([semproc, '_digitized', '_', digitization])
        visproc = ''.join([visproc, '_digitized', '_', digitization])

    # model architecture
    modelreps        = ['sem'] 
    modelinps        = ['phn', 'vwd'] 
    modelproc        = 'hub'
    modelarch        = f"{''.join(modelinps)}2{''.join(modelreps)}" 
    

    if 'sem' in modelreps:
        hub_dim = sem_dim
    if 'vis' in modelreps:
        hub_dim = vis_dim
    if 'wrd' in modelreps:
        hub_dim = wrd_dim

    # the hub dimension matches the vwd
    if not nvwd is None:
        hub_dim *= nvwd

    settings['bias'] = False   
    
    if 'phn' in modelinps:
        modelparams = f'_PIHID{phn_ihid_dim}L{phn_ihid_nlayers}D{phn_ihid_ndirs}'
    if modelproc == 'hub':
        modelparams += f'_HUB{hub_dim}L{hub_nlayers}D{hub_ndirs}'

    modelparams += '_PI' + str(ptimeincs) + "_VI" + str(vtimeincs)

    if segchars != 'none':
        modelparams += f'_SC{len(segchars)}'

    if tgt_full == 1:
        modelparams += '_TFULL'

    settings['lexicon']    = f'lexicon_W{nwords}'
    settings['wrdfname']   = f'lexicons/{wrdlist}/words/lexicon_W{nwords}.csv'
    settings['catfname']   = f'lexicons/{wrdlist}/words/categories_W{nwords}.csv'
    settings['semfname']   = f'lexicons/{wrdlist}/word/semantic_W{nwords}'\
                             f'_{semsource} _{semproc}.csv'
    settings['visfname']   = f'lexicons/{wrdlist}/word/visual_W{nwords}'\
                             f'_{vissource}_{visproc}.csv'
    if not trlfname is None:
        settings['trlfname']   = f'lexicons/{wrdlist}/words/traintrls_{trlfname}.csv'
    else:
        settings['trlfname']   = None
    
    # assigns outputs
    settings['nwords']    = nwords
    settings['wrdlist']   = wrdlist
    settings['phn_dim']   = phn_dim
    settings['wrd_dim']   = wrd_dim
    settings['sem_dim']   = sem_dim
    settings['semsource'] = semsource
    settings['semproc']   = semproc
    settings['vis_dim']   = vis_dim
    settings['vissource'] = vissource
    settings['visproc']   = visproc
    settings['vissteps']  = vissteps

    settings['modelparams']      = modelparams
    settings['modelreps']        = modelreps
    settings['modelinps']        = modelinps
    settings['modelproc']        = modelproc
    settings['modelarch']        = modelarch
    settings['nvwd']             = nvwd

    return settings
