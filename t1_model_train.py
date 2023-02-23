import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from t0_train_data_gen import generate_pairs, dataset_generation, zero_padded_adjmat
import config
import numpy as np
from sklearn.metrics import roc_curve,auc
import time

class cfg_embedding_layer(layers.Layer): # 单层CFG嵌入
    def __init__(self):
        super(cfg_embedding_layer,self).__init__()

    def build(self, input_shape):
        # self.theta = self.add_weight(name="P0", shape=tf.TensorShape([config.embedding_size, config.embedding_size]))  # 64*64
        self.theta = tf.Variable(tf.random.normal([config.config.embedding_size, config.config.embedding_size],mean = 0, stddev=0.1, dtype=tf.float32))
        # self.theta1 = self.add_weight(name="P1", shape=tf.TensorShape([config.embedding_size, config.embedding_size])) # 64*64
        self.theta1 = tf.Variable(tf.random.normal([config.config.embedding_size, config.config.embedding_size],mean = 0, stddev=0.1, dtype=tf.float32))
        self.theta = tf.nn.dropout(self.theta, rate = 0.9, seed = 1)
        self.theta1 = tf.nn.dropout(self.theta1, rate = 0.9, seed = 1)

        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=False, trainable=True, name='norm_layer')
        super(cfg_embedding_layer,self).build(input_shape)

    def call(self,input):
        '''
        :param input:shape = (batch,config.embedding_size,nodes)
        :return:
        '''
        if config.config.norm_layer_flag: # 层归一化设置
            curr_embedding = self.norm_layer(input)
            curr_embedding = tf.einsum('ik,akj->aij',self.theta,curr_embedding)
        else:
            curr_embedding = tf.einsum('ik,akj->aij',self.theta,input) # [64*64]P0参数 * [a*64*j]输入CFG
        curr_embedding = tf.nn.relu(curr_embedding) # 激活函数relu：ReLU(P0参数*输入CFG)
        curr_embedding = tf.einsum('ik,akj->aij',self.theta1,curr_embedding) # P1参数*ReLU(P0参数*输入CFG)
        # curr_embedding = tf.nn.relu(curr_embedding)
        return curr_embedding

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape


class dfg_embedding_layer(layers.Layer): # 单层DFG嵌入，结构同单层CFG嵌入
    def __init__(self):
        super(dfg_embedding_layer,self).__init__()

    def build(self, input_shape):
        self.theta = self.add_weight(name="Q0",shape=tf.TensorShape([config.config.embedding_size,config.config.embedding_size]))
        self.theta1 = self.add_weight(name="Q1",shape=tf.TensorShape([config.config.embedding_size,config.config.embedding_size]))
        self.theta = tf.nn.dropout(self.theta, rate = 0.9, seed = 1)
        self.theta1 = tf.nn.dropout(self.theta1, rate = 0.9, seed = 1)

        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=False, trainable=True, name='norm_layer')
        super(dfg_embedding_layer,self).build(input_shape)

    def call(self,input):
        '''
        :param input:shape = (batch,config.embedding_size,nodes)
        :return:
        '''
        if config.norm_layer_flag:
            curr_embedding = self.norm_layer(input)
            curr_embedding = tf.einsum('ik,akj->aij',self.theta,curr_embedding)
        else:
            curr_embedding = tf.einsum('ik,akj->aij',self.theta,input)
        curr_embedding = tf.nn.relu(curr_embedding)
        curr_embedding = tf.einsum('ik,akj->aij',self.theta1,curr_embedding)
        # curr_embedding = tf.nn.relu(curr_embedding)
        return curr_embedding

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape


def compute_graph_embedding(cfg_adjmat,dfg_adjmat,feature_mat,W1,W2,cfg_embed_layer,dfg_embed_layer): 
    '''
    构建整个网络，即以RNN结构组织上述CFG嵌入层，以及DFG嵌入层
    cfg_adjmat: shape = (batch,config.max_nodes,config.max_nodes)
    dfg_adjmat: shape = (batch,config.max_nodes,config.max_nodes)
    feature_mat: shape = (batch,config.max_nodes,feature_size)
    W1: shape = (config.embedding_size,feature_size)
    W2: shape = (config.embedding_size,config.embedding_size)
    '''
    feature_mat = tf.einsum('aij->aji',feature_mat) #
    # 自选特征矩阵.T
    '''
    初始嵌入层（单层参数并未初始化，或初始化参数为0#
    '''
    init_embedding = tf.zeros(shape=(config.max_nodes,config.embedding_size)) # 初始嵌入层，起降维作用？升维
    cfg_prev_embedding = tf.einsum('aik,kj->aij', cfg_adjmat, init_embedding)   
    cfg_prev_embedding = tf.einsum('aij->aji', cfg_prev_embedding)   
    dfg_prev_embedding = tf.einsum('aik,kj->aij',dfg_adjmat,init_embedding)  
    dfg_prev_embedding = tf.einsum('aij->aji',dfg_prev_embedding)  

    '''
    RNN迭代各层
    '''
    for iter in range(config.T): # RNN迭代 （previous 和 current区分
        cfg_neighbor_embedding = cfg_embed_layer(cfg_prev_embedding)   # CFG嵌入层 （类的用法？
        dfg_neighbor_embedding = dfg_embed_layer(dfg_prev_embedding)  # DFG嵌入层
        term = tf.einsum('ik,akj->aij', W1, feature_mat)   # 自选特征嵌入: W1*fea
        curr_embedding = tf.nn.tanh(term + cfg_neighbor_embedding + dfg_neighbor_embedding) # 该层求和且激活函数为tanh
        prev_embedding = curr_embedding  #  
        prev_embedding = tf.einsum('aij->aji',prev_embedding)  
        cfg_prev_embedding = tf.einsum('aik,akj->aij',cfg_adjmat,prev_embedding)   
        cfg_prev_embedding = tf.einsum('aij->aji',cfg_prev_embedding) 
        dfg_prev_embedding = tf.einsum('aik,akj->aij',dfg_adjmat,prev_embedding)
        dfg_prev_embedding = tf.einsum('aij->aji',dfg_prev_embedding)
    graph_embedding = tf.reduce_sum(curr_embedding,axis=2)  # 最终求和降维
    graph_embedding = tf.einsum('ij->ji',graph_embedding) # 转置
    graph_embedding = tf.matmul(W2,graph_embedding) # W2参数(config.embedding_size,config.embedding_size)*(j, i)
    return graph_embedding


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()  # 继承超类
        self.cfg_embed_layer = cfg_embedding_layer()
        self.dfg_embed_layer = dfg_embedding_layer()
        self.W1 = tf.Variable(tf.random.normal([config.embedding_size, config.vulseeker_feature_size],mean = 0, stddev=0.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.normal([config.embedding_size, config.embedding_size],mean = 0 ,stddev=0.1, dtype=tf.float32))
        # tf.
        # W1, W2初始化，平均分布，亦可用tf.random.normal正态分布

    def call(self, inputs, training=None, mask=None): # 所谓孪生网络
        g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat = inputs
        g1_embedding = compute_graph_embedding(g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,self.W1,self.W2,self.cfg_embed_layer,self.dfg_embed_layer)
        g2_embedding = compute_graph_embedding(g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,self.W1,self.W2,self.cfg_embed_layer,self.dfg_embed_layer)
        sim_score = cosine(g1_embedding, g2_embedding)  # 
        return sim_score,g1_embedding,g2_embedding


def cosine(q, a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q),axis=0))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a),axis=0))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(q,a), axis=0)
    score = tf.divide(pooled_mul_12, pooled_len_1 * pooled_len_2 +0.0001, name="scores")
    return score


def loss(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y):
    """
    Get the model's output(two graph's embeddings and their similarity),return the loss.
    :param model:
    :param g1_cfg_adjmat:
    :param g1_dfg_adjmat:
    :param g1_featmat:
    :param g2_cfg_adjmat:
    :param g2_dfg_adjmat:
    :param g2_featmat:
    :param y:
    :return:
    """
    input = (g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat)
    sim,g1_embedding,g2_embedding = model(input) # sim_score
    if tf.reduce_max(sim)>1 or tf.reduce_min(sim)<-1:
        sim = sim * 0.999  # Here because the float num computation can overflow,such as 1.00000001.
    loss_value = tf.reduce_sum(tf.square(tf.subtract(sim,y))) # 与标签值的欧氏距离
    return loss_value,sim,g1_embedding,g2_embedding


def grad(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y):
    with tf.GradientTape() as tape:
        loss_value,sim,g1_embedding,g2_embedding = loss(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
    return loss_value,tape.gradient(loss_value,model.trainable_variables),sim,g1_embedding,g2_embedding
    # 返回loss值，参数更新？，相似值

def train():
    actual_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.learning_rate,decay_steps=config.decay_steps, \
                                                                            decay_rate=config.decay_rate)
    optimizer = tf.optimizers.Adam(actual_learning_rate)  # 基于Adam算法的优化器类
    # learning_rate_dbg = optimizer.lr
    model = MyModel()  # MyModel类
    model.build([(None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.vulseeker_feature_size),
                (None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.vulseeker_feature_size)])
    model.summary()
    max_auc = 0
    train_loss =[]
    valid_loss = []
    train_auc = []
    valid_auc = []
    train_accuracy = []
    valid_accuracy = []
    # 开始训练
    for epoch in range(config.epochs): # 100个config.epochs
        start = time.time()
        pos_count = 0
        neg_count = 0
        train_dataset = dataset_generation() # 每个epoch随机采样1000个，给150
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.BinaryAccuracy()
        epoch_auc_avg = tf.keras.metrics.AUC(num_thresholds=498)
        step = 0
        for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in train_dataset:
            # 输入归一化
            if config.norm_flag:
                g1_mean,g1_var = tf.nn.moments(g1_featmat, axes = [0, 1, 2]) # axes=[0] 沿0轴简单归一化，此处为全局归一化
                g1_featmat_shape = tf.shape(g1_featmat)
                scale = tf.Variable(tf.ones(shape = g1_featmat_shape)) 
                shift = tf.Variable(tf.zeros(shape = g1_featmat_shape))
                # g1_mean = shift
                epsilon = 1e-3
                g1_featmat = tf.nn.batch_normalization(g1_featmat, g1_mean, g1_var, shift, scale, epsilon)

                g2_mean,g2_var = tf.nn.moments(g2_featmat, axes = [0, 1, 2])
                g2_featmat_shape = tf.shape(g2_featmat)
                scale = tf.Variable(tf.ones(shape = g2_featmat_shape)) 
                shift = tf.Variable(tf.zeros(shape = g2_featmat_shape))
                # g2_mean = shift
                g2_featmat = tf.nn.batch_normalization(g2_featmat, g2_mean, g2_var, shift, scale, epsilon)

                g1_cfg_adjmat = g1_cfg_adjmat*2-1
                g1_dfg_adjmat = g1_dfg_adjmat*2-1
                g2_cfg_adjmat = g2_cfg_adjmat*2-1
                g2_dfg_adjmat = g2_dfg_adjmat*2-1
            loss_value,grads,sim,_,_ = grad(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            epoch_loss_avg(loss_value)
            sim = (sim.numpy()+1)/2
            y = (y.numpy()+1)/2
            epoch_accuracy_avg.update_state(y,sim)
            epoch_auc_avg.update_state(y,sim)
            auc_avg_res = epoch_auc_avg.result().numpy()
            if step%(config.train_step_per_epoch//100)==0:  # step为150的整数倍，每个step为一个batch，即10个样本
                print("step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step,epoch_loss_avg.result(),epoch_accuracy_avg.result(),auc_avg_res))
                if step==100*(config.train_step_per_epoch//100): # 一个epoch，15000steps,150000样本
                    break
            pos_count +=  sum(y==1)
            neg_count +=  sum(y==0)
        end = time.time()
        train_loss.append(epoch_loss_avg.result())
        train_accuracy.append(epoch_accuracy_avg.result())
        train_auc.append(epoch_auc_avg.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(epoch,epoch_loss_avg.result(),epoch_accuracy_avg.result(),epoch_auc_avg.result()))
        print('time:'+str(end-start))
        v_loss,v_accuracy,v_auc = valid(model)
        valid_loss.append(v_loss)
        valid_accuracy.append(v_accuracy)
        valid_auc.append(v_auc)
        end = time.time()
        print('time:'+str(end-start))
        if v_auc > max_auc:
            model.save(config.T1_VULSEEKER_MODEL_TO_SAVE, save_format='tf')
            max_auc = v_auc
    # 测试
    fpr, tpr, test_auc = test(model)
    # 作图
############################################
    x = range(config.epochs)
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.title("Accuracy curve")
    plt.plot(x, train_accuracy, label="train_accuracy")
    plt.plot(x, valid_accuracy, label="valid_accuracy")
    plt.xlabel("config.epochs")
    plt.ylabel("accuracy")
    plt.legend()

    plt.subplot(222)
    plt.title("AUC curve")
    plt.plot(x, train_auc, label="train_auc")
    plt.plot(x, valid_auc, label="valid_auc")
    plt.xlabel("config.epochs")
    plt.ylabel("AUC")
    plt.legend()

    plt.subplot(223)
    plt.title("Loss curve")
    
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, valid_loss, label="valid_loss")
    plt.xlabel("config.epochs")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(224)
    plt.plot(fpr,tpr,  label="test set(AUC:"+ str(test_auc) + ")")
    plt.title("test set ROC")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    plt.savefig(config.T1_VULSEEKER_FIGURE_TO_SAVE + "results.png")
    plt.show()
    print("----------------------------")

def valid(model):
    valid_dataset = dataset_generation(type="valid")
    epoch_loss_avg_valid = tf.keras.metrics.Mean()
    epoch_accuracy_avg_valid = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_avg = tf.keras.metrics.AUC()
    step = 0
    print("-------------------------")
    for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in valid_dataset:
        # 输入归一化
        if config.norm_flag:
            g1_mean,g1_var = tf.nn.moments(g1_featmat, axes = [0, 1, 2]) # axes=[0] 沿0轴简单归一化，此处为全局归一化
            g1_featmat_shape = tf.shape(g1_featmat)
            scale = tf.Variable(tf.ones(shape = g1_featmat_shape)) 
            shift = tf.Variable(tf.zeros(shape = g1_featmat_shape))
            # g1_mean = shift
            epsilon = 1e-3
            g1_featmat = tf.nn.batch_normalization(g1_featmat, g1_mean, g1_var, shift, scale, epsilon)

            g2_mean,g2_var = tf.nn.moments(g2_featmat, axes = [0, 1, 2])
            g2_featmat_shape = tf.shape(g2_featmat)
            scale = tf.Variable(tf.ones(shape = g2_featmat_shape)) 
            shift = tf.Variable(tf.zeros(shape = g2_featmat_shape))
            # g2_mean = shift
            g2_featmat = tf.nn.batch_normalization(g2_featmat, g2_mean, g2_var, shift, scale, epsilon)

            g1_cfg_adjmat = g1_cfg_adjmat*2-1
            g1_dfg_adjmat = g1_dfg_adjmat*2-1
            g2_cfg_adjmat = g2_cfg_adjmat*2-1
            g2_dfg_adjmat = g2_dfg_adjmat*2-1
        loss_value, grads, sim, _, _ = grad(model, g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
        # 没有optimizer，验证集上无参数反向传播
        sim = (sim.numpy()+1)/2
        y = (y.numpy()+1)/2
        epoch_loss_avg_valid(loss_value)
        epoch_accuracy_avg_valid.update_state(y, sim)
        epoch_auc_avg.update_state(y,sim)

        if step % (config.valid_step_per_epoch//10) == 0: 
            print("valid step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg_valid.result(),
                                                                       epoch_accuracy_avg_valid.result(),epoch_auc_avg.result()))
            if step == 10*(config.valid_step_per_epoch//10):  
                break
        step += 1
    print("----------------------------")
    return epoch_loss_avg_valid.result(),epoch_accuracy_avg_valid.result(),epoch_auc_avg.result()


def test():
    model = tf.keras.models.load_model(config.T1_VULSEEKER_MODEL_TO_SAVE)
    epoch_loss_avg_test = tf.keras.metrics.Mean()
    epoch_accuracy_avg_test = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_sim = []
    epoch_auc_ytrue = []
    test_dataset = dataset_generation(type="test")
    step = 0
    for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in test_dataset:
    # 输入归一化
        if config.norm_flag:
            g1_mean,g1_var = tf.nn.moments(g1_featmat, axes = [0, 1, 2]) # axes=[0] 沿0轴简单归一化，此处为全局归一化
            g1_featmat_shape = tf.shape(g1_featmat)
            scale = tf.Variable(tf.ones(shape = g1_featmat_shape)) 
            shift = tf.Variable(tf.zeros(shape = g1_featmat_shape))
            # g1_mean = shift
            epsilon = 1e-3
            g1_featmat = tf.nn.batch_normalization(g1_featmat, g1_mean, g1_var, shift, scale, epsilon)

            g2_mean,g2_var = tf.nn.moments(g2_featmat, axes = [0, 1, 2])
            g2_featmat_shape = tf.shape(g2_featmat)
            scale = tf.Variable(tf.ones(shape = g2_featmat_shape)) 
            shift = tf.Variable(tf.zeros(shape = g2_featmat_shape))
            # g2_mean = shift
            g2_featmat = tf.nn.batch_normalization(g2_featmat, g2_mean, g2_var, shift, scale, epsilon)

            g1_cfg_adjmat = g1_cfg_adjmat*2-1
            g1_dfg_adjmat = g1_dfg_adjmat*2-1
            g2_cfg_adjmat = g2_cfg_adjmat*2-1
            g2_dfg_adjmat = g2_dfg_adjmat*2-1
        loss_value, grads, sim, _, _ = grad(model, g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
        epoch_loss_avg_test(loss_value)
        sim = (sim.numpy()+1)/2
        y = (y.numpy()+1)/2
        epoch_accuracy_avg_test.update_state(y, sim)

        epoch_auc_sim = np.concatenate((epoch_auc_sim,sim))
        epoch_auc_ytrue = np.concatenate((epoch_auc_ytrue,y))
        fpr,tpr,thres = roc_curve(epoch_auc_ytrue,epoch_auc_sim,pos_label=1)
        auc_score = auc(fpr,tpr)
        if step % (config.test_step_per_epoch//10) == 0:
            print("test step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg_test.result(),epoch_accuracy_avg_test.result(),auc_score))
            if step == 10*(config.test_step_per_epoch//10):
                break
        step += 1
    return fpr, tpr, auc_score

if __name__ == "__main__":
    test_flag = 0
    if not test_flag:
        train()
    else:
        test()
