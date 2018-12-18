max_epochs = 50
batch_size = 256
keep_prob = 0.75
image_dimension = 1024
hidden_dimension = embed_dimension = 512


class CaptionConfig(object):
    layers = 2
    model_name = 'model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d' % (
        keep_prob, batch_size, hidden_dimension, embed_dimension, layers)


class ModelConfig(object):
    layers = 3
    model_name = 'model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d' % (
        keep_prob, batch_size, hidden_dimension, embed_dimension, layers)
