"""TFX House Pricing preprocessing.

This file defines a template for TFX Transform component.
"""

import tensorflow as tf
import tensorflow_transform as tft

from models import features

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}
    
    for key in features.DENSE_FLOAT_FEATURE_KEYS:
        outputs[features.transformed_name(key)] = tft.scale_to_z_score(inputs[key])
        
    for key in features.VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
        inputs[key],
        top_k=features.VOCAB_SIZE,
        num_oov_buckets=features.OOV_SIZE)
        
    for key, num_buckets in zip(features.BUCKET_FEATURE_KEYS,
                              features.BUCKET_FEATURE_BUCKET_COUNT):
        outputs[features.transformed_name(key)] = tft.bucketize(inputs[key], num_buckets)
    
    outputs[features.transformed_name(features.LABEL_KEY)] = inputs[features.LABEL_KEY]
    
    return outputs
        
    

    