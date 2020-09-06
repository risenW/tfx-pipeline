"""TFX House Pricing Model Features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""


# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = ['bedrooms',
                             'bathrooms',
                             'sqft_living',
                             'sqft_lot',
                             'floors',
                             'waterfront',
                             'view',
                             'condition',
                             'sqft_above',
                             'sqft_basement',
                             'yr_built',
                             'yr_renovated']


# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
# BUCKET_FEATURE_KEYS = ['yr_built','yr_renovated']

# # Number of buckets used by tf.transform for encoding each feature. The length
# # of this list should be the same with BUCKET_FEATURE_KEYS.
# BUCKET_FEATURE_BUCKET_COUNT = [12, 12]

# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
CATEGORICAL_FEATURE_KEYS = ['city', 'statezip']

# # Number of buckets to use integer numbers as categorical features.
CATEGORICAL_FEATURE_MAX_VALUES = [32, 24]

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = ["street"]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000


# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10


# Target Key
LABEL_KEY = 'price'


def transformed_name(key):
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'

def vocabulary_name(key):
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'

def transformed_names(keys):
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]

