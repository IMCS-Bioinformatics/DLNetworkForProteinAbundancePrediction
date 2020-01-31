import tensorflow as tf
from tensorflow.python.ops import standard_ops
from utils.eq_layer import variance_scaling_lr
regul = 0.0001

class protein_model:
    def __init__(self, n_dims, n_types, n_genes, n_rp):
        self.n_types = n_types
        self.n_dims = n_dims
        self.n_genes = n_genes
        self.rna_value = tf.placeholder("float32", [None])
        self.dim_in = tf.placeholder("float32", [None, self.n_dims])
        self.rp_in = tf.placeholder("float32", [None, n_rp])
        self.type_in = tf.placeholder("int32", [None])
        self.plength_in = tf.placeholder("float32", [None])
        self.gene_id = tf.placeholder("int32", [None])
        self.gtex_value = tf.placeholder("float32", [None])
        self.optimizer = None
        self.train_loss = None
        #self.test_loss = None
        self.n_middle = 20
        self.saturation_costs = []  # here we will collect all saturation costs
        self.saturation_weight = 1e-5
        self.weight_inference = tf.Variable(False, trainable=False)
        self.mul_weight = None
        self.add_weight = None

    def relu(self, x):
        result = tf.nn.relu(x)
        #result = tf.nn.sigmoid(x)
        dif = result-x
        self.saturation_costs.append(tf.reduce_mean(dif))
        return result

    def addNoise(self, d, doTrain):
        #d = tf.cond(self.is_training > 0, lambda: d*tf.random_uniform(tf.shape(d), minval=0.5, maxval=1.5),lambda: d)
        #d = tf.cond(self.is_training > 0, lambda: d * tf.truncated_normal(tf.shape(d), mean = 1.0, stddev=0.05), lambda: d)
        # d = tf.cond(self.is_training > 0,
        #             lambda: d + tf.truncated_normal(tf.shape(d), stddev=0.1),
        #             lambda: d)

        #d = tf.cond(self.is_training > 0, lambda: d * tf.truncated_normal(tf.shape(d), mean=1.0, stddev=0.2), lambda: d)
        #if doTrain: d = d * tf.exp(tf.truncated_normal(tf.shape(d), mean=0.0, stddev=0.1))
        if doTrain: d = d * tf.truncated_normal(tf.shape(d), mean=1.0, stddev=0.7)
        #if doTrain: d = d + tf.truncated_normal(tf.shape(d), mean=0.0, stddev=0.01)
        #d = tf.cond(self.is_training > 0, lambda: tf.nn.dropout(d, self.dropout_keep_prob), lambda: d)

        return d

    def dense(self, input, units,name,reuse, kernel_regularizer, initZero=False):
        shape = input.get_shape().as_list()
        input_dim = shape[-1]

        with tf.variable_scope(name, reuse=reuse):
            initializer = tf.constant_initializer(0.0) if initZero else tf.truncated_normal_initializer(stddev=1.0)
            kernel = tf.get_variable('kernel',shape=[input_dim, units], initializer=initializer,regularizer=kernel_regularizer)
            bias = tf.get_variable('bias', shape=[units], initializer=tf.constant_initializer(0.0))
            scale = variance_scaling_lr([input_dim, units], 'FAN_AVG')
            kernel=kernel*scale
        output = standard_ops.matmul(input, kernel)
        output = tf.nn.bias_add(output, bias)
        return output

    def createResult(self, in_batch, type_batch, dims_batch, gene_id, reuse = False, doTrain = True):
        in_1 = tf.expand_dims(in_batch, axis=1)
        if self.n_types>0:
            type_hot = tf.one_hot(type_batch, self.n_types)
        else: type_hot = tf.ones([tf.shape(dims_batch)[0],1]) #do not use type if testset_by_type
        type_hot = tf.cond(self.weight_inference, lambda:type_hot*0,lambda :type_hot)
        if doTrain: dims_batch = tf.nn.dropout(dims_batch, 0.75)
        #if doTrain: dims_batch = self.addNoise(dims_batch, doTrain)
        #type_1 = self.dense(type_hot, units=80, name="middleType", reuse=reuse, kernel_regularizer=None)
        in_f = self.dense(dims_batch, units=80, name="middleF", reuse=reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(regul), initZero=True)
        #if doTrain: in_f = self.addNoise(in_f, doTrain)
        all_inputs = tf.concat([in_f, type_hot, self.rp_in], 1)
        #all_inputs = in_f*type_1

        if self.n_genes>0:
            with tf.variable_scope("embed", reuse=reuse):
                embedding_weights = tf.get_variable(
                    "embedding", [self.n_genes, 20], dtype=tf.float32, initializer=tf.constant_initializer(1.0))#, regularizer=tf.contrib.layers.l2_regularizer(regul)
                scale = variance_scaling_lr([1, 20], 'FAN_AVG')
                embedding_weights = embedding_weights*scale

                in_gene_id = tf.nn.embedding_lookup(embedding_weights, gene_id)
                in_gene_id = tf.cond(self.weight_inference, lambda: in_gene_id * 0, lambda: in_gene_id)
                all_inputs = tf.concat([all_inputs,in_gene_id], 1)

        features = self.dense(all_inputs, units=80, name="middlexx", reuse=reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(regul))

        features = tf.cond(self.weight_inference, lambda:features, lambda:tf.nn.relu(features))
        if doTrain: features = tf.nn.dropout(features, 0.9)

        #features = tf.concat([features, tf.expand_dims(self.plength_in, axis=1)], 1)

        add_weight = self.dense(features, units=1, name="addWeight", reuse = reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(regul))
        mul_weight = self.dense(features, units=1, name="mulWeight", reuse = reuse,kernel_regularizer=tf.contrib.layers.l2_regularizer(regul))
        #mul_weight1 = self.dense(features, units=1, name="mulWeight1", reuse=reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(regul))
        #mul_weight = self.dense(type_hot, units=1, name="mulWeight", reuse=reuse, kernel_regularizer=tf.contrib.layers.l2_regularizer(regul))

        result = in_1*mul_weight+add_weight
        #result = tf.log(tf.maximum(1e-4, result + 1))

        return result, mul_weight, add_weight

    def createResultGraph(self):
        self.result, self.mul_weight, self.add_weight = self.createResult(self.rna_value, self.type_in, self.dim_in, self.gene_id, reuse = True, doTrain = False)

    def createTrainGraph(self):
        """Creates graph for training"""
        result, _, _ = self.createResult(self.rna_value, self.type_in, self.dim_in, self.gene_id)

        out_1 = tf.expand_dims(self.gtex_value, axis=1)
        dif = tf.squared_difference(result, out_1)
        #dif = tf.abs(result- out_1)

        self.train_loss = tf.reduce_mean(dif)
        #self.train_loss = tf.reduce_sum(dif/out_1)/tf.reduce_sum(1.0/out_1)

        #self.loss = tf.add(tf.py_func(my_func, [mul_weight], [tf.float32]), 0)

        # optimizer
        cost = self.train_loss
        if len(self.saturation_costs)>0:
            sat_cost = tf.add_n(self.saturation_costs) / len(self.saturation_costs)
            cost += sat_cost*self.saturation_weight
        
        reg_losses = tf.convert_to_tensor(sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), dtype=tf.float32)
        cost+= reg_losses
        #optimizer = tf.contrib.opt.LazyAdamOptimizer(0.0001, beta1=0.9)
        optimizer = tf.train.AdamOptimizer(0.0003, beta1=0.9)
        #optimizer = tf.train.MomentumOptimizer(0.1,0.9, use_nesterov=False)# orig 0.0005
        #optimizer = tf.train.GradientDescentOptimizer(0.0001)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        self.optimizer = optimizer.minimize(cost)
        #self.train_loss = cost
        #self.train_loss = tf.sqrt(self.train_loss) # for quadratic loss

    def getResult(self, sess, dims, type, in_val, p_lengths, gene_ids, rp_data):
        res = sess.run(self.result, feed_dict={self.rna_value: in_val, self.dim_in:dims, self.type_in:type, self.plength_in:p_lengths, self.gene_id:gene_ids, self.rp_in:rp_data})
        return res

    def getResult_ext(self, sess, dims, type, in_val, p_lengths, gene_ids, rp_data):
        res = sess.run([self.result, self.mul_weight, self.add_weight], feed_dict={self.rna_value: in_val, self.dim_in:dims, self.type_in:type, self.plength_in:p_lengths, self.gene_id:gene_ids, self.rp_in:rp_data})
        return res

    def getWeights(self, sess, dims, type, p_lengths, rp_data):
        gene_ids = [0]
        addw, mulw = sess.run([self.add_weight, self.mul_weight], feed_dict={self.dim_in:dims, self.type_in:type,self.plength_in:p_lengths, self.gene_id:gene_ids, self.rp_in:rp_data})
        return addw, mulw

    def train(self, sess, dims, type, in_val, out_val, p_lengths, gene_ids, rp_data):
        """do training"""

        res, loss = sess.run([self.optimizer, self.train_loss],
                             feed_dict={self.rna_value: in_val, self.dim_in:dims, self.type_in:type, self.gtex_value:out_val, self.plength_in:p_lengths, self.gene_id:gene_ids, self.rp_in:rp_data})

        return loss
