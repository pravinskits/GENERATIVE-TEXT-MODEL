from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import textwrap

# === GPT-2 SETUP ===
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
model_gpt2.eval()

def generate_text_gpt2(prompt, max_length=100):
    inputs = tokenizer_gpt2.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model_gpt2.generate(inputs, max_length=max_length, do_sample=True, temperature=0.7)
    return tokenizer_gpt2.decode(outputs[0], skip_special_tokens)

gpt2_prompt = "The future of space exploration is"
gpt2_generated = generate_text_gpt2(gpt2_prompt)

# === LSTM SETUP ===
corpus = [
    "Artificial intelligence is transforming industries",
    "Machine learning is a field of artificial intelligence",
    "Deep learning uses neural networks to solve problems",
    "Natural language processing helps machines understand human speech"
]

tokenizer_lstm = Tokenizer()
tokenizer_lstm.fit_on_texts(corpus)
total_words = len(tokenizer_lstm.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer_lstm.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = np.eye(total_words)[y]

model_lstm = Sequential()
model_lstm.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(total_words, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X, y, epochs=300, verbose=0)

def generate_text_lstm(seed_text, next_words=15):
    for _ in range(next_words):
        token_list = tokenizer_lstm.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model_lstm.predict(token_list, verbose=0)
        output_word = tokenizer_lstm.index_word.get(np.argmax(predicted), '')
        seed_text += ' ' + output_word
    return seed_text

lstm_generated = generate_text_lstm("Artificial intelligence", 15)

# === CREATE OUTPUT IMAGE ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
wrapped_gpt2 = "\n".join(textwrap.wrap("GPT-2 Output: " + gpt2_generated, 90))
wrapped_lstm = "\n".join(textwrap.wrap("LSTM Output: " + lstm_generated, 90))
output_text = wrapped_gpt2 + "\n\n" + wrapped_lstm
ax.text(0.01, 0.99, output_text, fontsize=12, va="top", ha="left")
plt.tight_layout()
plt.savefig("text_generation_output.png")
plt.show()
