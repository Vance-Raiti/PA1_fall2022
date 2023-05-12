'''SGD implementation'''

from variable import Variable



def SGD(loss_fn, params, data, learning_rate):
    ''''performs an SGD update and returns updated parameters.
    arguments:
        loss_fn: function that takes params, data and returns a loss value.
        params: list of Variables representing parameters of some model.
        data: list of Variables representing minibatch data.
        learning_rate: learning rate for SGD.
    returns: two values:
        new_params: Variable containing next value for params after SGD update,
        correct: float number that is 1.0 if the prediction was correct, and 
            0.0 otherwise.
    '''

    ### YOUR CODE HERE ###
    
    
    for d in data:
        loss, correct = loss_fn(d)
        loss.backward()
    
    for param in params:
        param.data -= param.grad*learning_rate
        param.grad = None
    new_params = params

    return new_params, correct







