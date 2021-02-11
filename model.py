import tensorflow as tf
import numpy as np
import pandas as pd


class Cox_autoencoder(tf.keras.Model):
    def __init__(self,
                 feature_dim,
                 l1=True,
                 l2=False,
                 coefficient=0.01,
                 drop_rate=0.1,
                 hidden_dim=256,
                 deep_dim=16,
                 activation=tf.nn.relu):
        super(Cox_autoencoder, self).__init__()
        tf.keras.backend.set_floatx('float64')
        self.l1 = tf.keras.regularizers.l1(l=coefficient)
        self.l2 = tf.keras.regularizers.l2(l=coefficient)
        self.l1l2 = tf.keras.regularizers.l1_l2(l1=coefficient, l2=coefficient)
        if l1 and l2:
            self.reg = self.l1l2
        else:
            if l1:
                self.reg = self.l1
            elif l2:
                self.reg = self.l2
            else:
                self.reg = None
        self.dense1 = tf.keras.layers.Dense(hidden_dim,
                                            activation=activation,
                                            kernel_regularizer=self.reg)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.bottleneck = tf.keras.layers.Dense(deep_dim,
                                                activation=activation,
                                                kernel_regularizer=self.reg)
        self.norm = tf.keras.layers.BatchNormalization()
        self.cox_regression = tf.keras.layers.Dense(1,
                                                    kernel_regularizer=self.reg)
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation=activation,
                                            kernel_regularizer=self.l1)
        self.dense_final = tf.keras.layers.Dense(feature_dim)

    def call(self, input_x, input_y, training=None):
        # encoder
        event_batch = np.array(input_y[:, 0])
        event_batch = tf.convert_to_tensor(event_batch, dtype=tf.float64)

        r_batch = np.array([[int(x >= y) for x in input_y[:, 1]] for y in input_y[:, 1]])
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float64)
        # encoder
        x = self.dropout(input_x)
        x = self.dense1(x)
        x = self.dropout(x)
        deep_feature = self.bottleneck(x)
        # cox regression
        # deep_feature = self.norm(x)
        deep_feature = self.dropout(deep_feature)
        hazard = self.cox_regression(deep_feature)
        # decoder
        x = self.dense2(deep_feature)
        output = self.dense_final(x)
        # print("hazard", hazard.numpy())
        return output, deep_feature, hazard, event_batch, r_batch


class Cox_nnet(tf.keras.Model):
    def __init__(self,
                 l1=True,
                 l2=False,
                 coefficient=0.01,
                 drop_rate=0.1,
                 hidden_dim=256,
                 deep_dim=16,
                 activation=tf.nn.relu):
        super(Cox_nnet, self).__init__()
        tf.keras.backend.set_floatx('float64')
        self.l1 = tf.keras.regularizers.l1(l=coefficient)
        self.l2 = tf.keras.regularizers.l2(l=coefficient)
        self.l1l2 = tf.keras.regularizers.l1_l2(l1=coefficient, l2=coefficient)
        if l1 and l2:
            self.reg = self.l1l2
        else:
            if l1:
                self.reg = self.l1
            elif l2:
                self.reg = self.l2
            else:
                self.reg = None
        self.dense1 = tf.keras.layers.Dense(hidden_dim,
                                            activation=activation,
                                            kernel_regularizer=self.reg)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.bottleneck = tf.keras.layers.Dense(deep_dim,
                                                activation=activation,
                                                kernel_regularizer=self.reg)
        self.cox_regression = tf.keras.layers.Dense(1,
                                                    kernel_regularizer=self.reg)

    def call(self, input_x, input_y, training=None):
        event_batch = np.array(input_y[:, 0])
        event_batch = tf.convert_to_tensor(event_batch, dtype=tf.float64)

        r_batch = np.array([[int(x >= y) for x in input_y[:, 1]] for y in input_y[:, 1]])
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float64)
        # neural network
        x = self.dropout(input_x)
        x = self.dense1(x)
        x = self.dropout(x)
        deep_feature = self.bottleneck(x)
        deep_feature = self.dropout(deep_feature)
        hazard = self.cox_regression(deep_feature)
        return deep_feature, hazard, event_batch, r_batch


class Cox(tf.keras.Model):
    """
    class of cox, initiating parameters l1,l2 for lasso and ridge,respectively.
    """

    def __init__(self,
                 l1=False,
                 l2=False,
                 coefficient=0.01,
                 drop_rate=0.1):
        super(Cox, self).__init__()
        tf.keras.backend.set_floatx('float64')
        self.l1 = tf.keras.regularizers.l1(l=coefficient)
        self.l2 = tf.keras.regularizers.l2(l=coefficient)
        self.l1l2 = tf.keras.regularizers.l1_l2(l1=coefficient, l2=coefficient)

        if l1 and l2:
            self.reg = self.l1l2
        else:
            if l1:
                self.reg = self.l1
            elif l2:
                self.reg = self.l2
            else:
                self.reg = None

        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.cox_regression = tf.keras.layers.Dense(1,
                                                    kernel_regularizer=self.reg)

    def call(self, input_x, input_y, training=None):
        event_batch = np.array(input_y[:, 0])
        r_batch = np.array([[int(x >= y) for x in input_y[:, 1]] for y in input_y[:, 1]])
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float64)
        # cox regression layer
        x = self.dropout(input_x)
        hazard = self.cox_regression(x)
        return hazard, event_batch, r_batch


def normalization(x):
    """"
    :param x: numpy array
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def get_loss(input_x, hazard, event_batch, r_batch, *args):
    """loss function for all the models"""
    # risk set
    # hazard = normalization(hazard)
    risk_set = tf.matmul(r_batch, tf.math.exp(hazard))
    risk_set = tf.math.log(risk_set)
    pl_loss = -tf.reduce_sum((hazard - risk_set) * event_batch) / tf.reduce_sum(event_batch)
    # for cox_ae model
    if len(args) == 2:
        # print("args", args)
        output = args[0]  # output of autoencoder
        coefficient = args[1]  # weight of pl_loss
        ae_loss = tf.reduce_mean(tf.square(output - input_x),
                                 name="ae_loss")
        all_loss = ae_loss + coefficient * pl_loss
        return pl_loss, ae_loss, all_loss

    return pl_loss


def get_grad_cox_ae(model, input_x, input_y):
    with tf.GradientTape() as tape:
        output, deep_feature, hazard, event_batch, r_batch = model(input_x, input_y)
        pl_loss, ae_loss, all_loss = get_loss(input_x,
                                              hazard,
                                              event_batch,
                                              r_batch,
                                              output,
                                              1)
        grads = tape.gradient([all_loss, ae_loss, pl_loss],
                              model.trainable_variables)
    return pl_loss, ae_loss, all_loss, grads, deep_feature


def get_grad_cox_nnet(model, input_x, input_y):
    with tf.GradientTape() as tape:
        deep_feature, hazard, event_batch, r_batch = model(input_x, input_y)
        pl_loss = get_loss(input_x, hazard, event_batch, r_batch)
        grads = tape.gradient([pl_loss], model.trainable_variables)
    return pl_loss, grads, deep_feature


def get_grad_cox(model, input_x, input_y):
    with tf.GradientTape() as tape:
        hazard, event_batch, r_batch = model(input_x, input_y)
        pl_loss = get_loss(input_x, hazard, event_batch, r_batch)
        grads = tape.gradient([pl_loss], model.trainable_variables)
    return pl_loss, grads


def train_model(model,
                x_train,
                y_train,
                batch_size=512,
                global_step=0,
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                num_epochs=1000):
    # 确定stop条件
    for epoch in range(num_epochs):
        for x in range(0, len(x_train), batch_size):
            global_step += 1
            x_inp = x_train[x: x + batch_size]
            y_inp = y_train[x: x + batch_size]
            if isinstance(model, Cox_autoencoder):
                print("cox_ae instance")
                pl_loss, ae_loss, all_loss, grads, deep_feature = get_grad_cox_ae(model,
                                                                                  x_inp,
                                                                                  y_inp)
            elif isinstance(model, Cox_nnet):
                print("cox_nnet instance")
                pl_loss, grads, deep_feature = get_grad_cox_nnet(model,
                                                                 x_inp,
                                                                 y_inp)
            elif isinstance(model, Cox):
                print("cox instance")
                pl_loss, grads = get_grad_cox(model, x_inp, y_inp)
            else:
                print("Please give the right model!")
                return ()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("epoch:", epoch + 1)
            print("global_step:", global_step)
            print("pl_loss:", pl_loss.numpy())
            if isinstance(model, Cox_autoencoder):
                print("ae_loss:", ae_loss.numpy())
                print("all_loss:", all_loss.numpy())

    return model


def get_deep_feature_hazard_df(deep_feature, hazard, index):
    """
    :param deep_feature:tensor from ae_cox regression model
    :param hazard:tensor of prognosis index
    :param index:sample name
    :return dataframe
    """
    # batch = deep_feature.shape[0]
    dim = deep_feature.shape[1]
    deep_feature_hazard_df = pd.DataFrame(index=index)
    deep_feature_hazard_df["sample"] = deep_feature_hazard_df.index
    for i in range(dim):
        col_name = "deep_feature" + str(i + 1)
        deep_feature_hazard_df[col_name] = deep_feature.numpy()[..., i]  # 第i+1维数据
    deep_feature_hazard_df["prognosis_index"] = hazard.numpy()
    return deep_feature_hazard_df


def get_hazard_df(hazard, index):
    """
    :param hazard:tensor of prognosis index
    :param index:sample name
    :return dataframe
    """
    hazard_df = pd.DataFrame(index=index)
    hazard_df["sample"] = hazard_df.index
    hazard_df["prognosis_index"] = hazard.numpy()
    return hazard_df
