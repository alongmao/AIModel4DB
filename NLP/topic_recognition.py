#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Modify Time 2023/8/6 00:04
@Author maoalong  
@Version 1.0
@Desciption None
"""

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFBertModel

topicModel = tf.keras.models.load_model("/home/along/AI-models/nlp_model/post/saved_dict/bert_fine_tuned.h5",
                                        custom_objects={"TFBertModel": TFBertModel})
tokenizer = AutoTokenizer.from_pretrained('/home/along/AI-models/nlp_model/post/bert-base-cased')
cat_encoder = LabelEncoder()
category_to_categoryID_mapping = {'ARTS & CULTURE': 0, 'BUSINESS': 1, 'COLLEGE': 2, 'COMEDY': 3, 'CRIME': 4,
                                  'DIVORCE': 5, 'EDUCATION': 6, 'ENTERTAINMENT': 7, 'ENVIRONMENT': 8, 'FIFTY': 9,
                                  'FOOD & DRINK': 10, 'GOOD NEWS': 11, 'HOME & LIVING': 12, 'IMPACT': 13, 'MEDIA': 14,
                                  'PARENTS': 15, 'POLITICS': 16, 'QUEER VOICES': 17, 'RELIGION': 18, 'SCIENCE': 19,
                                  'SPORTS': 20, 'STYLE & BEATUY': 21, 'STYLE & BEAUTY': 22, 'TECH': 23, 'TRAVEL': 24,
                                  'U.S. NEWS': 25, 'VOICES': 26, 'WEDDINGS': 27, 'WEIRD NEWS': 28, 'WELLNESS': 29,
                                  'WOMEN': 30, 'WORLD NEWS': 31}
categoryIdD_to_category_mapping = {category_to_categoryID_mapping[key]: key for key in
                                   category_to_categoryID_mapping.keys()}
cat_encoder.classes_ = list(category_to_categoryID_mapping.keys())

max_len = 256  # Same as used during training


def extract_topic(texts):
    encode_text = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
    )
    predicted = topicModel.predict(
        {'input_ids': encode_text['input_ids'], 'attention_mask': encode_text['attention_mask']})
    predicted_category_ids = tf.argmax(predicted, axis=1).numpy()
    response = []
    for (text, predict_id) in zip(texts, predicted_category_ids):
        response.append(
            {"text": text, "topic_id": int(predict_id), "topic": categoryIdD_to_category_mapping[predict_id]})
    return response
