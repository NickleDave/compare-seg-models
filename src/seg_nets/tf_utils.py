import functools
import tensorflow as tf


# doublewrap and define_scope functions taken from blog post by Danijar Hafner
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
# https://danijar.github.io/structuring-your-tensorflow-models
def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# spatial_dropout taken from:
# https://stats.stackexchange.com/questions/282282/
# how-is-spatial-dropout-in-2d-implemented
def spatial_dropout(input,
                    rate,
                    seed):
    """spatial dropout layer

    Parameters
    ----------
    input : arr
        tensor with shape B x W x H x F
        where F is the number of feature maps for that layer
    rate : float
        proportion of feature maps we want to keep
    seed : int
        seed for random number generator
    """

    # get the batch size and number of feature maps
    num_feature_maps = [tf.shape(input)[0], tf.shape(input)[3]]

    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = rate
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=input.dtype)

    # if we take the floor of this, we get a binary matrix where
    # (1-keep_prob)% of the values are 0 and the rest are 1
    binary_tensor = tf.floor(random_tensor)

    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, 1, tf.shape(input)[3]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(input, rate) * binary_tensor
    return ret