import tensorflow as tf
import tensorlayer as tl


def preprocess_deconv_output_shape(x, shape):

    if shape[0] is None or shape[0] == -1:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = list(shape)
    return shape



class TransposeLayer(tl.layers.Layer):
    """
        The :class:`TransposeLayer` class is layer which reshape the tensor.

        Parameters
        ----------
        layer : a :class:`Layer` instance
            The `Layer` class feeding into this layer.
        perm : a list
            The dimension shuffler.
        name : a string or None
            An optional name to attach to this layer.

        """

    def __init__(
            self,
            layer=None,
            perm=[],
            name='transpose_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = tf.transpose(self.inputs, perm=perm, name=name)
        print("  tensorlayer:Instantiate TransposeLayer %s: %s" % (self.name, self.outputs._shape))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class ConvRNNLayer(tl.layers.Layer):
    """
    The :class:`RNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_shape : tuple, the shape of each cell width*height
    filter_size : tuple, the size of filter width*height
    cell_fn : a TensorFlow's core Convolutional RNN cell as follow.
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    feature_map : a int
        The number of feature map in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    return_last : boolen
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    """
    def __init__(
        self,
        layer = None,
        cell_shape = None,
        feature_map = 1,
        filter_size = (3, 3),
        cell_fn = None,
        cell_init_args = {},
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        n_steps=5,
        initial_state = None,
        return_last = False,
        return_seq_2d = False,
        name='convlstm_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate RNNLayer %s: feature_map:%d, n_steps:%d, "
              "in_dim:%d %s, cell_fn:%s " % (self.name, feature_map,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 5 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(5)
        except:
            raise Exception("RNN : Input dimension should be rank 5 : [batch_size, n_steps, input_x, "
                            "input_y, feature_map]")



        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        self.cell = cell = cell_fn(shape=cell_shape, filter_size=filter_size, num_features=feature_map)
        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 1.2.3
        state = self.initial_state
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :, :, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print(" n_params : %d" % (len(rnn_variables)))

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = outputs[-1]
        else:
            if return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, cell_shape[0]*cell_shape[1]*feature_map])
            else:
                # <akara>: stack more RNN layer after that
                # 5D Tensor [n_example/n_steps, n_steps, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, cell_shape[0],
                                                                  cell_shape[1], feature_map])

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )


class BiConvRNNLayer(tl.layers.Layer):
    """
    The :class:`BiRNNLayer` class is a Bidirectional RNN layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    fw_initial_state : None or forward RNN State
        If None, initial_state is zero_state.
    bw_initial_state : None or backward RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_last : boolen
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    fw(bw)_final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    fw(bw)_initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`ReshapeLayer`.

    References
    ----------
    - `Source <https://github.com/akaraspt/deepsleep/blob/master/deepsleep/model.py>`_
    """

    def __init__(
        self,
        layer = None,
        cell_shape=None,
        feature_map=1,
        filter_size=(3, 3),
        cell_fn=None,
        cell_init_args = None,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        n_steps=5,
        fw_initial_state=None,
        bw_initial_state=None,
        dropout=None,
        n_layer=1,
        return_last=False,
        return_seq_2d=False,
        name = 'biconvrnn_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate BiRNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, "
              "cell_fn:%s, dropout:%s, n_layer:%d " % (self.name, feature_map,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            self.batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % self.batch_size)
        else:
            from tensorflow.python.ops import array_ops
            self.batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(5)
        except:
            raise Exception("RNN : Input dimension should be rank 5 : [batch_size, n_steps, n_features]")

        with tf.variable_scope(name, initializer=initializer) as vs:
            self.fw_cell = cell_fn(shape=cell_shape, filter_size=filter_size, num_features=feature_map)
            self.bw_cell = cell_fn(shape=cell_shape, filter_size=filter_size, num_features=feature_map)
            # Apply dropout
            if dropout:
                if type(dropout) in [tuple, list]:
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]
                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout
                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                    "float)")
                self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                          self.fw_cell,
                          input_keep_prob=in_keep_prob,
                          output_keep_prob=out_keep_prob)
                self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                          self.bw_cell,
                          input_keep_prob=in_keep_prob,
                          output_keep_prob=out_keep_prob)
            # Apply multiple layers
            if n_layer > 1:
                print("     n_layer: %d" % n_layer)
                try:
                    self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell] * n_layer,
                                                          state_is_tuple=True)
                    self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell] * n_layer,
                                                          state_is_tuple=True)
                except:
                    self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell] * n_layer)
                    self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell] * n_layer)

            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # exit()
            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unpack(self.inputs, axis=1)
            outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if return_last:
                self.outputs = outputs[-1]
            else:
                self.outputs = outputs
                if return_seq_2d:
                    # 2D Tensor [n_example, n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, n_steps*cell_shape[0]*cell_shape[1],
                                                                           feature_map*2])
                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, cell_shape[0],
                                                                      cell_shape[1], feature_map*2])
            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

            # Retrieve just the RNN variables.
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )


class DuplicateLayer(tl.layers.Layer):

    """ The :class:`ConcatLayer` class is layer which concat (merge)
    two or more :class:`DenseLayer` to a single class:`DenseLayer`.
    Parameters ----------
    layer : class:`Layer` instances The `Layer` class feeding into this layer.
    dup_num : int Dimension along which to duplicate.
    name : a string or None An optional name to attach to this layer.
    dup_num: The number of duplication.
    """

    def __init__(
            self,
            layer=None,
            dup_num=1,
            name='duplicate_layer',
            pack_dim=1
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.input = layer.outputs
        self.outputs = []
        self.dup_num = dup_num
        for l in xrange(self.dup_num):
            self.outputs.append(layer.outputs)
        self.outputs = tf.pack(self.outputs, axis=pack_dim, name=name)  # 1.2

        print("  tensorlayer:Instantiate DuplicateLayer %s, %d" % (self.name, self.dup_num))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class DeConv3dLayer(tl.layers.Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer,
    see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`_.
    Parameters
    ----------
    layer : a :class:`Layer` instance The `Layer` class feeding into this layer.
    act : activation function The function that is applied to the layer activations.
    shape : list of shape shape of the filters, [depth, height, width, output_channels, in_channels],
    filter's in_channels dimension must match that of value. output_shape : list of output shape
    representing the output shape of the deconvolution op. strides : a list of ints.
    The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID". The type of padding algorithm to use.
    W_init : weights initializer The initializer for initializing the weight matrix.
    b_init : biases initializer The initializer for initializing the bias vector.
    W_init_args : dictionary The arguments for the weights initializer.
    b_init_args : dictionary The arguments for the biases initializer.
    name : a string or None An optional name to attach to this layer. """

    def __init__( self, layer = None, act = tf.identity, shape = [2, 2, 2, 128, 256],
                  output_shape = [1, 12, 32, 32, 128], strides = [1, 2, 2, 2, 1], padding='SAME',
                  W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
                  W_init_args={}, b_init_args={}, name ='decnn3d_layer', ):

        tl.layers.Layer.__init__(self, name=name)

        self.inputs = layer.outputs
        print(" [TL] DeConv3dLayer %s: shape:%s out_shape:%s strides:%s pad:%s act:%s" %
              (self.name, str(shape), str(output_shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_deconv3d', shape=shape, initializer=W_init, **W_init_args )
            b = tf.get_variable(name='b_deconv3d', shape=(shape[-2]), initializer=b_init, **b_init_args)

            output_shape = preprocess_deconv_output_shape(self.inputs, output_shape)

            self.outputs = act(tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape,
                strides=strides, padding=padding) + b )
            self.outputs = tf.reshape(self.outputs, output_shape)
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
            self.all_layers.extend( [self.outputs] )
            self.all_params.extend( [W, b] )

