import numpy as np
import math

def tokenize(text):
    return text.lower().split()


def build_vocab(tokens):
    vocab = {}
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def embed_tokens(token_ids, vocab_size, d_model, seed=0):
    # embedding table
    rng = np.random.default_rng(seed)
    embedding_matrix = rng.normal(0, 1, size=(vocab_size, d_model))
    return embedding_matrix[token_ids]


def positional_encoding(seq_len, d_model):
    """
    sinusoidal PE from AIAYN section 3.5
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # for pos in range(seq_len):
    #     for i in range(d_model):
    #         if i % 2 == 0:
    #             pe[pos][i] = math.sin(pos / 10000**(i / d_model))
    #         else:
    #             pe[pos][i] = math.cos(pos / 10000**((i-1) / d_model))

    pe = np.zeros((seq_len, d_model))

    pos = np.arange(seq_len)[:, None]       # shape (seq_len, 1)
    i = np.arange(d_model)[None, :]         # shape (1, d_model)
    angle_rate = 1 / 10000 ** ((2 * (i // 2)) / d_model)
    angles = pos * angle_rate               # shape (seq_len, d_model) <- due to broadcasting

    pe[:, 0::2] = np.sin(angles[:, 0::2])   # all even indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # all odd indices
    return pe


if __name__ == "__main__":
    text = "the cat sat on the mat"
    tokens = tokenize(text)
    vocab = build_vocab(tokens)
    token_ids = np.array([vocab[t] for t in tokens])

    d_model = 8
    embeddings = embed_tokens(token_ids, len(vocab), d_model)
    pe = positional_encoding(len(tokens), d_model)

    final_input = embeddings + pe
    print(final_input.shape)
    print(final_input)