import os


def describe(x, description='', level='brief'):
    '''
    Print out information about a variable.
    Args:
        x (any): the variable to describe
        description (string): text to describe the variable - default empty
        level (string): description detail level - 'values', 'brief' (default)
    Returns:
        none
    '''
    if isinstance(x, list):
        if description:
            print(description)
        print(f'list with {len(x)} {type(x[0])} elements')
        if(level == 'values'):
            print('Values: \n{}'.format(x))
    else:
        if description:
            print(description)
        print(f'{x.shape} array')
        if(level == 'values'):
            print('Values: \n{}'.format(x))


def describe_tensor(x, description='', level='brief'):
    '''
    Print out information about a tensor

    Args:
        x (torch Tensor): the tensor to describe
        description (string): text to describe the tensor - default empty
        level (string): description details level - 'values', 'brief' (default)
    Returns:
        none
    '''
    if description:
        print(description)
    print('Type  : {}'.format(x.type()))
    print('Shape : {}'.format(x.shape))
    if(level == 'values'):
        print('Values: \n{}'.format(x))


def describe_model(model, description='', level='brief'):
    '''
    Print out information about a model

    Args:
        model (nn.Model): the model to describe
        description (string): text to describe the variable - default empty
        level (string): description detail level - 'values', 'brief' (default)
    Returns:
        none
    '''
    print('')
    if description:
        print(description)
    for name, param in model.named_parameters():
        if param.requires_grad:
            describe(param.data, name, level)


def create_folder(path, verbose=True):
    """
        Creates a folder if it does not exist.

        Args:
            path (string): path to required folder
            verbose (bool): print out info or warnings (default: True)
        Returns:
            path (string): the path or '' deoending on status
            status (bool): whether the action could be completed or not
    """
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            if verbose:
                print (f'Failed to create folder {path}')
            return '', False
        else:
            if verbose:
                print (f'Successfully created folder {path}')
            return path, True
    else:
        if verbose:
            print(f'Folder {path} already exists, content will be overwritten')
        return path, True

    
def create_folder(path, verbose=True):
    """
        Creates a folder if it does not exist.
        
        Args:
            path (string): path to required folder
            verbose (bool): print out info or warnings (default: True)
        Returns:
            path (string): the path or '' deoending on status
            status (bool): whether the action could be completed or not
    """
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            if verbose:
                print (f'Failed to create folder {path}')
            return '', False
        else:
            if verbose:
                print (f'Successfully created folder {path}')
            return path, True
    else:
        if verbose:
            print(f'Folder {path} already exists, content will be overwritten')
        return path, True