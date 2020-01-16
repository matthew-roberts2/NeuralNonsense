import numpy as np
import tensorflow as tf
import os
import random

tf.enable_eager_execution()


def load_names():
    with open('./processed/all_names.txt') as f:
        return [l.strip().split(',')[0].lower() for l in f]


def keras_preprocess(data):
    return tf.keras.preprocessing.sequence.pad_sequences(data, padding='post')


def normalize_names(names):
    longest = len(max(names, key=len))
    return ['{}{}'.format(name, ''.join([' ' for _ in range(longest - len(name))])) for name in names]


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate(model, start_string, char_map, idx_map, size=4, temp=1.0, min=4):
    text = []
    temperature = temp
    num_generate = size

    input_eval = [char_map[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, dim=1)

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        p_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([p_id], 0)
        text.append(idx_map[p_id])

    while len(text) < min:
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        p_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([p_id], 0)
        text.append(idx_map[p_id])

    return start_string + ''.join(text)


def main():
    names = load_names()
    vocab = [' '] + sorted(set(''.join([name for name in names])))
    # names = keras_preprocess(names)  # normalize_names(names)

    skip_build_model = False
    batch_size = 256
    examples_per_epoch = 1000
    steps_per_epoch = examples_per_epoch
    buffer_size = 10000
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 64
    epochs = 15
    checkpoint_dir = './model/namegen'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

    char_index = {u: i for i, u in enumerate(vocab)}
    index_char = np.array(vocab)
    vec_names = keras_preprocess([[char_index[c] for c in name] for name in names])

    if not skip_build_model:
        char_dataset = tf.data.Dataset.from_tensor_slices(vec_names)

        def shift(chunk):
            inn_t = chunk[:-1]
            out_t = chunk[1:]
            return inn_t, out_t

        dataset = char_dataset.map(shift)
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
        model.compile(tf.train.AdamOptimizer(), loss=loss)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
        model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    # Generation suite
    for _ in range(20):
        print(generate(model, index_char[random.randint(1, 26)], char_index, index_char, 15, 1.0))

    # Dump a bunch to a file, just to observe
    with open('net_completion_dump.txt', 'w') as f:
        for i in range(500):
            if i % 100 == 0:
                print(i)
            f.write(f'{generate(model, index_char[random.randint(1, 26)], char_index, index_char, 15, 1.0)}\n')


if __name__ == "__main__":
    main()
