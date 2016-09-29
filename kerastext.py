"""
tools to do text classification with keras
"""

from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.base import ClassifierMixin
from keras.preprocessing import sequence
from collections import Counter
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Lambda, Input, merge, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping
import keras.backend as K
    

def f1_score(y, y_pred):
    beta = 1
    y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
    num_true = K.sum(y)
    num_pred = K.sum(y_pred_binary)
    tp = K.sum(y * y_pred_binary)
    recall = K.switch(num_true>0, tp / num_true, 0)
    precision = K.switch(num_pred>0, tp / num_pred, 0)
    precision_recall_sum = recall + (beta*precision)
    return K.switch(precision_recall_sum>0, (beta+1)*((precision*recall)/(precision_recall_sum)), 0)

# def precision(y, y_pred):
#     y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
#     num_true = K.sum(y)
#     num_pred = K.sum(y_pred_binary)
#     tp = K.sum(y * y_pred_binary)
#     return K.switch(num_pred>0, tp / num_pred, 0)


def f4_score(y, y_pred):
    beta = 4
    y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
    num_true = K.sum(y)
    num_pred = K.sum(y_pred_binary)
    tp = K.sum(y * y_pred_binary)
    recall = K.switch(num_true>0, tp / num_true, 0)
    precision = K.switch(num_pred>0, tp / num_pred, 0)
    precision_recall_sum = recall + (beta*precision)
    return K.switch(precision_recall_sum>0, (beta+1)*((precision*recall)/(precision_recall_sum)), 0)
    
from theano.ifelse import ifelse

def num_true(y, y_pred):
#     target_recall = 24
#     num_true = y.nonzero()[0].shape[0]
#     target_tp_t = T.iround(target_recall * num_true)
    return y[10][0]

def target_tp_t(y, y_pred):
    target_recall = 0.95
    num_true = y.nonzero()[0].shape[0]
    return num_true

    
def precision_at_recall(y, y_pred):
    target_recall = 0.95
    num_true = y.nonzero()[0].shape[0]
    target_tp_t = T.iround(target_recall * num_true)    
    pos_inds = y.nonzero()[0]
    pred_pos = y_pred[pos_inds]
    p_argsort = pred_pos.argsort(axis=0)
    pred_cutoff = ifelse(num_true > 0, y_pred[pos_inds[p_argsort[-target_tp_t]]][0][0], np.float32(0))
    return precision_at_cutoff(y, y_pred, pred_cutoff)

from theano.printing import Print as Tpr

# def spec_at_sens(y, y_pred):
#     target_recall = 0.95
#     pred_sort = y_pred.argsort(axis=0)
#     pos_inds = yp.nonzero()[0]
    
    
    
#     num_true = Tpr('num_true')(pos_inds.shape[0])
#     target_tp_t = Tpr("target_tp_t")(T.iround(target_recall * num_true))
#     pred_pos = Tpr("pred_pos")(y_predp[pos_inds])
#     p_argsort = Tpr("p_argsort")(pred_pos.argsort(axis=0))
# #     cutoff_position_in_pos = p_argsort[-target_tp_t]    
# #     cutoff_position_in_all = pos_inds[cutoff_position_in_pos]
# #     cutoff_pred = y_predp[cutoff_position_in_all]

#     some_are_true = Tpr('some_are_true')(T.gt(num_true, 0))
#     pred_cutoff = ifelse(some_are_true, y_predp[pos_inds[p_argsort[-target_tp_t]]][0][0], np.float32(0))
#     return specificity_at_cutoff(yp, y_predp, pred_cutoff)


def spec_at_sens2(y, y_pred):
    target_recall = 0.90
    # extend both by one value
    y_e = T.concatenate([np.array([1]), y.T[0]])
    y_pred_e = T.concatenate([np.array([0.]), y_pred.T[0]])
    
    pos_inds = y_e.nonzero()[0]
    num_true = pos_inds.shape[0]
    target_tp_t = ifelse(num_true > 1, T.iround(target_recall * (num_true-1)) + 1, np.int64(0))
    
    pred_pos = y_pred_e[pos_inds]
    p_argsort = pred_pos.argsort(axis=0)
    
    pred_cutoff = y_pred_e[pos_inds[p_argsort[-target_tp_t]]]
    
    return specificity_at_cutoff(y, y_pred, pred_cutoff)    
    
    



def specificity_at_cutoff(y, y_pred, pred_cutoff):
    y_pred_binary = T.switch(y_pred >= pred_cutoff, np.int64(1), np.int64(0))
    tn = T.eq(y + y_pred_binary, 0).nonzero()[0].shape[0]
    num_neg = T.eq(y, 0).nonzero()[0].shape[0]
    return T.switch(num_neg>0, tn/num_neg, np.float32(0))

def precision_at_cutoff(y, y_pred, pred_cutoff):
    y_pred_binary = T.switch(y_pred >= pred_cutoff, np.int64(1), np.int64(0))
    num_pred = y_pred_binary.nonzero()[0].shape[0]
    tp_cutoff = T.eq(y + y_pred_binary, 2).nonzero()[0].shape[0]
    precision = T.switch(num_pred>0, tp_cutoff / num_pred, 0)
    return precision

def precision(y, y_pred):
    return precision_at_cutoff(y, y_pred, np.float32(0.5))

def specificity(y, y_pred):
    return specificity_at_cutoff(y, y_pred, np.float32(0.5))

def recall_at_cutoff(y, y_pred, pred_cutoff):
    y_pred_binary = T.switch(y_pred >= pred_cutoff, np.int64(1), np.int64(0))
    tp = T.eq(y + y_pred_binary, 2).nonzero()[0].shape[0]
    num_true = y.nonzero()[0].shape[0]
    return T.switch(num_true>0, tp/num_true, np.float32(0))
    
def recall(y, y_pred):
    return recall_at_cutoff(y, y_pred, np.float(0.5))

def cutoff_score(y, y_pred):
    target_recall = 0.95
    pos_inds = y.nonzero()[0]
    pred_pos = y_pred[pos_inds]
    p_argsort = pred_pos.argsort(axis=0)    
    target_tp_t = T.iround(target_recall * T.sum(y))
    pred_cutoff = ifelse(pos_inds.shape[0] > 0, y_pred[pos_inds[p_argsort[-target_tp_t]]][0][0], np.float32(0))
    return pred_cutoff
class KerasVectorizer(VectorizerMixin):    
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 analyzer='word', embedding_inits=None):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = (1, 1)
        self.embedding_inits = embedding_inits # optional gensim word2vec model
        self.embedding_dim = embedding_inits.syn0.shape[1] if embedding_inits else None

    def fit(self, raw_documents):
        """
        sets up the word -> int mapping, where 0=most
        """
        analyzer = self.build_analyzer()
        word_counter = Counter(analyzer(' '.join(raw_documents))).most_common()
        # 0 = pad, 1 = start, 2 = OOV
        self.vocab_map = {w[0]: i+3 for i, w in enumerate(word_counter)}         
        if self.embedding_inits:
            rand_vec = lambda: np.random.rand(self.embedding_dim)*2-1
            weights_words = [self.embedding_inits[w[0]] if w[0] in self.embedding_inits else rand_vec() for w in word_counter]
            weights_special = [rand_vec() for i in range(3)]
            self.embedding_weights = np.array(weights_special + weights_words)
        
    def transform(self, raw_documents):
        """
        returns lists of integers
        """
        analyzer = self.build_analyzer()
        int_lists = [[1]+[self.vocab_map.get(w, 2) for w in analyzer(t)] for t in raw_documents]
        # 0 = pad, 1 = start, 2 = OOV
#         import pdb; pdb.set_trace()
        return sequence.pad_sequences(int_lists)
    
    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)        

class CNNTextClassifier(ClassifierMixin):
    
    def __init__(self, max_features=10000, max_len=400, batch_size=50,
                 stopping_patience=5, dropout=0.5, activation='relu',
                 num_filters=100, filter_size=None, num_epochs=10,
                 num_hidden_layers=0, dim_hidden_layers=200,
                 stopping_target='val_f4_score', stopping_less_is_good=True,
                 embedding_dim=200, embedding_weights=None, optimizer='adam',
                 undersample_ratio=None, oversample_ratio=None, class_weight=None,
                 validation_split=0,
                 l2=3):
        # default hyperparams taken from Kim paper
        self.max_features = max_features
        self.max_len = max_len
        self.batch_size = batch_size
        self.stopping_patience = stopping_patience
        self.dropout = dropout
        self.activation = activation
        self.num_filters = num_filters        
        self.filter_size = [3, 4, 5] if filter_size is None else filter_size
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.dim_hidden_layers = dim_hidden_layers
        self.stopping_target = stopping_target
        self.stopping_mode = "min" if stopping_less_is_good else "max"
        self.embedding_dim = embedding_dim
        self.embedding_weights = self.get_embedding_weights(embedding_weights)
        self.optimizer = optimizer
        self.model = self.generate_model()
        self.undersample_ratio = undersample_ratio # for class imbalance with few positive examples
        self.oversample_ratio = oversample_ratio
        self.class_weight = class_weight
        self.validation_split = validation_split

        
    def fit(self, X_train, y_train):
        print("Processing data ({} samples)".format(len(y_train)))
        X_train = self.low_pass_filter(X_train)
        
        if self.validation_split:
            X_train, X_val, y_train, y_val = self.get_val_set(X_train, y_train) # skim a bit off for monitoring
            self.validation_data = (X_val, y_val)
        else:
            self.validation_data = None
        if self.undersample_ratio:
            X_train, y_train = self.undersample(X_train, y_train, self.undersample_ratio)
            print("Sampled with ratio of {}, reduced to {} samples.".format(self.undersample_ratio, len(y_train)))            
        elif self.oversample_ratio:
            X_train, y_train = self.oversample(X_train, y_train, self.oversample_ratio)
            print("Sampled with ratio of {}, increased to {} samples.".format(self.oversample_ratio, len(y_train)))        
        if self.stopping_patience:
            callbacks = [EarlyStopping(monitor=self.stopping_target, patience=self.stopping_patience, verbose=0, mode=self.stopping_mode)]
        else:
            callbacks = []            
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=self.num_epochs,
                       verbose=1, class_weight=self.class_weight,
                       validation_data=self.validation_data, callbacks=callbacks)
        
        # debug - retry metrics manually
        
#         print("Predicting...")
#         self.preds = self.predict(self.validation_data[0]['input'])
#         pred_tensor = theano.shared(self.preds.astype(np.float32))
#         true_tensor = theano.shared(self.validation_data[1]['output'].astype(np.float32))
        
#         print("Scores: precision {} recall {} precision @ recall {}".format(K.eval(precision(true_tensor, pred_tensor)), K.eval(recall(true_tensor, pred_tensor)), K.eval(precision_at_recall(true_tensor, pred_tensor))))
        
        
        
#     def fit_resample(self, X_train, y_train):
#         X_train = self.low_pass_filter(X_train)
#         X_t, X_val, y_t, y_val = self.get_val_set(X_train, y_train) # skim a bit off for monitoring
#         for epoch in range(self.num_epochs):
#             X_t_s, y_t_s = self.subsample(X_t, y_t, self.sampling_ratio)
#             self.model.fit({"input":X_t_s}, {"output":y_t_s}, batch_size=self.batch_size, nb_epoch=1,
#                            verbose=1, class_weight=self.class_weight,
#                            validation_data=({"input":X_val}, {"output":y_val}))
            
    def predict(self, X_test):
        X_test = self.low_pass_filter(X_test)
        return self.model.predict({"input":X_test})
    
    def low_pass_filter(self, X):
        # 1. set maximum document length
        X=X[:,-self.max_len:]
        # 2. remove words less frequent than self.max_features
        X[X>self.max_features] = 2
        return X
    
    def get_embedding_weights(self, emb):
        if emb is None:
            return None
        else:
            return [emb[:self.max_features+3]]
        
    def undersample(self, X_train, y_train, ratio):
        """
        sample a proportion of negative samples for class imbalanced problems
        """
        y_bool = y_train.astype('bool')
        pos_indices = np.where(y_bool==True)[0]
        neg_indices = np.where(y_bool==False)[0]
        sampled_indices = (np.append(pos_indices, np.random.choice(neg_indices, int(len(pos_indices)*ratio), replace=False)))
        print("{} sampled indices from {} total, which comprise {} positive, {} negative examples".format(len(sampled_indices), len(y_bool), len(pos_indices), int(len(pos_indices)*ratio)))
        return X_train[sampled_indices], y_train[sampled_indices]
    
    def oversample(self, X_train, y_train, ratio):
        """
        oversample positive samples for class imbalanced problems
        """
        y_bool = y_train.astype('bool')
        pos_indices = np.where(y_bool==True)[0]
        neg_indices = np.where(y_bool==False)[0]
        sampled_indices = (np.append(neg_indices, np.random.choice(pos_indices, int(len(neg_indices)*ratio), replace=True)))
        print("{} sampled indices from {} total, which comprise {} positive, {} negative examples".format(len(sampled_indices), len(y_bool), ratio * len(neg_indices), len(neg_indices)))
        return X_train[sampled_indices], y_train[sampled_indices]


    def get_val_set(self, X, y):
        num_rows = X.shape[0]
#         inds = np.random.permutation(num_rows)

        cutoff = int(num_rows * (1-self.validation_split))
#         inds_val, inds_train = inds[:cutoff], inds[cutoff:]
#         return X[inds_train], X[inds_val], y[inds_train], y[inds_val]
        return X[:cutoff], X[cutoff:], y[:cutoff], y[cutoff:]
        
            
    def generate_model(self):
        k_inp = Input(shape=(self.max_len,), dtype='int32', name='input')
        k_emb = Embedding(input_dim=self.max_features+3, output_dim=self.embedding_dim,
                        input_length=self.max_len, weights=self.embedding_weights)(k_inp)

        k_conv_list = []
        for n in self.filter_size:
            k_conv = Convolution1D(nb_filter=self.num_filters,
                                    filter_length=n,
                                    border_mode='valid',
                                    activation='relu',
                                    subsample_length=1)(k_emb)
            
            k_maxpool1d = MaxPooling1D(pool_length=self.max_len - n + 1)(k_conv)            
            k_flatten = Flatten()(k_maxpool1d)
            k_conv_list.append(k_flatten)
            

        if len(k_conv_list)==1:
            k_merge = k_conv_list[0]
        else:
            k_merge = merge(k_conv_list, mode='concat', concat_axis=1)
            
        # add hidden layers if wanted
        last_dims = len(self.filter_size)*self.num_filters
        last_layer = k_merge
        
        if self.num_hidden_layers == 0:
        # put dropout after merge if no hidden layers
            last_layer = Dropout(self.dropout)(last_layer)

        for n in range(self.num_hidden_layers):
            k_dn = Dense(self.dim_hidden_layers, input_dim=last_dims, W_regularizer=l2(3))(last_layer)
            k_dp = Dropout(self.dropout)(k_dn)
            last_layer = Activation('relu')(k_dp)            
            last_dims = self.dim_hidden_layers
            
        k_dn = Dense(1, input_dim=last_dims)(last_layer)
        
        k_dp = Dropout(self.dropout)(k_dn)
            
        k_out = Activation('sigmoid', name="output")(k_dp)

        model = Model(input=[k_inp], output=[k_out])

        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer,
#                       metrics=['accuracy', num_true, target_tp_t, f1_score, precision, recall, specificity, spec_at_sens2, y_sum, y_ones, y_zeros, y_element,
#                               yp_sum, yp_mean, yp_element])
                      metrics=['accuracy', f1_score, precision, recall, specificity, spec_at_sens2])


        return model

