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
    
def f2_score(y, y_pred):
    beta = 2
    y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
    num_true = K.sum(y)
    num_pred = K.sum(y_pred_binary)
    tp = K.sum(y * y_pred_binary)

    recall = K.switch(num_true>0, tp / num_true, 0)
    precision = K.switch(num_pred>0, tp / num_pred, 0)

    precision_recall_sum = recall + (beta*precision)

    return K.switch(precision_recall_sum>0, (beta+1)*((precision*recall)/(precision_recall_sum)), 0)

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


def precision(y, y_pred):
    y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
    num_true = K.sum(y)
    num_pred = K.sum(y_pred_binary)
    tp = K.sum(y * y_pred_binary)
    return K.switch(num_pred>0, tp / num_pred, 0)

def recall(y, y_pred):
    y_pred_binary = K.round(y_pred) # > 0.5 goes to 1.0
    num_true = K.sum(y)
    num_pred = K.sum(y_pred_binary)
    tp = K.sum(y * y_pred_binary)
    return K.switch(num_true>0, tp / num_true, 0)



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
    
    def __init__(self, max_features=10000, max_len=500, batch_size=64,
                 stopping_patience=10, dropout=0.2, activation='relu',
                 num_filters=100, filter_size=None, num_epochs=40,
                 num_hidden_layers=1, dim_hidden_layers=200,
                 stopping_target='loss', stopping_less_is_good=True,
                 embedding_dim=200, embedding_weights=None, optimizer='adam',
                 sampling_ratio=None, class_weight=None, validataion_split=0.1):
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
        self.stopping_less_is_good = stopping_less_is_good
        self.embedding_dim = embedding_dim
        self.embedding_weights = self.get_embedding_weights(embedding_weights)
        self.optimizer = optimizer
        self.model = self.generate_model()
        self.sampling_ratio = sampling_ratio # for class imbalance with few positive examples
        self.class_weight = class_weight
        self.validation_split = validataion_split

        
    def fit(self, X_train, y_train):
        print("Processing data ({} samples)".format(len(y_train)))
        X_train = self.low_pass_filter(X_train)
        X_train, X_val, y_train, y_val = self.get_val_set(X_train, y_train) # skim a bit off for monitoring
        if self.sampling_ratio:
            X_train, y_train = self.subsample(X_train, y_train, self.sampling_ratio)
            print("Sampled with ratio of {}, reduced to {} samples.".format(self.sampling_ratio, len(y_train)))
        self.model.fit({"input":X_train}, {"output":y_train}, batch_size=self.batch_size, nb_epoch=self.num_epochs,
                       verbose=1, class_weight=self.class_weight,
                       validation_data=({"input":X_val}, {"output":y_val}))
        
    def fit_resample(self, X_train, y_train):
        X_train = self.low_pass_filter(X_train)
        X_t, X_val, y_t, y_val = self.get_val_set(X_train, y_train) # skim a bit off for monitoring
        for epoch in range(self.num_epochs):
            X_t_s, y_t_s = self.subsample(X_t, y_t, self.sampling_ratio)
#             import pdb; pdb.set_trace()
            self.model.fit({"input":X_t_s}, {"output":y_t_s}, batch_size=self.batch_size, nb_epoch=1,
                           verbose=1, class_weight=self.class_weight,
                           validation_data=({"input":X_val}, {"output":y_val}))
            
    def predict(self, X_test):

        X_test = self.low_pass_filter(X_test)
        return self.model.predict({"input":X_test})
    
    def low_pass_filter(self, X):
        # 1. set maximum document length
        X=X[:,:self.max_len]
        # 2. remove words less frequent than self.max_features
        X[X>self.max_features] = 2
        return X
    
    def get_embedding_weights(self, emb):
        if emb is None:
            return None
        else:
            return [ebm[:self.max_features+3]]
        
    def subsample(self, X_train, y_train, ratio):
        y_bool = y_train.astype('bool')
        pos_indices = np.where(y_bool==True)[0]
        neg_indices = np.where(y_bool==False)[0]
        sampled_indices = (np.append(pos_indices, np.random.choice(neg_indices, int(len(pos_indices)*ratio), replace=False)))
        print("{} sampled indices from {} total, which comprise {} positive, {} negative examples".format(len(sampled_indices), len(y_bool), len(pos_indices), int(len(pos_indices)*ratio)))
        return X_train[sampled_indices], y_train[sampled_indices]

    def get_val_set(self, X, y):
        num_rows = X.shape[0]
        inds = np.random.permutation(num_rows)
        cutoff = int(num_rows * self.validation_split)
        inds_val, inds_train = inds[:cutoff], inds[cutoff:]
        return X[inds_train], X[inds_val], y[inds_train], y[inds_val]
        
            
    def generate_model(self):
        inp = Input(shape=(self.max_len,), dtype='int32', name='input')
        emb = Embedding(output_dim=self.embedding_dim, input_dim=self.max_features+3, input_length=self.max_len, weights=self.embedding_weights)(inp)

        n_gram_filters = []
        for n in self.filter_size:
            n_gram_filters.append(Flatten()(MaxPooling1D(pool_length=self.max_len - n + 1)(Convolution1D(nb_filter=self.num_filters,
                                    filter_length=n,
                                    border_mode='valid',
                                    activation=self.activation,
                                    subsample_length=1)(emb))))

        if len(n_gram_filters)==1:
            merged_layer = n_gram_filters[0]
        else:
            merged_layer = merge(n_gram_filters, mode='concat', concat_axis=1)

        dp = Dropout(self.dropout)(merged_layer)
        # dn = Dense(1, input_dim=len(n_gram_filters)*nb_filter, W_regularizer=l2(l2_norm), activity_regularizer=activity_l2(l2_norm))(dp)

        dn = Dense(1, input_dim=len(self.filter_size)*self.num_filters)(dp)
        out = Activation('sigmoid', name="output")(dn)

        model = Model(input=[inp], output=[out])

        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy', precision, recall, f1_score])
        return model

