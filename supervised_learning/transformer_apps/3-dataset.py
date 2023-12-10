#1/usr/bin/env python3
'''task 1'''
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset():
    '''dataset class'''
    def __init__(self, batch_size, max_len):
        '''innit func'''
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self. tokenizer_en = self.tokenize_dataset(self.data_train)
        self.batch_size = batch_size
        self.max_len = max_len

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_length=self.max_len):
            '''fileter'''
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(2**15, reshuffle_each_iteration=True).padded_batch(self.batch_size)
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length).padded_batch(self.batch_size)



    def tokenize_dataset(self, data):
        '''toke dat data boiii'''
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, _ in data),
        target_vocab_size=2**15
        )

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for _, en in data),
        target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''encode'''
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)

    def tf_encode(self, pt, en):
        '''tf wrap'''
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens

