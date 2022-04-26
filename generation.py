import argparse
from nltk import tokenize
import streamlit as st
import string

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="Apple. Apple Event - March 8.")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)
parser.add_argument("--do_sample", default=True)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--num_return_sequences", default=2)
parser.add_argument("--num_sentences", default=2)
args = parser.parse_args()

def clear_output_text(text):
    text = text.replace("\\n", "")
    nonalpha = string.digits + string.punctuation + string.whitespace
    text = text.lstrip(nonalpha)
    filter = ''.join([chr(i) for i in range(1, 32)])
    text.translate(str.maketrans('', '', filter))
    text = text.replace('...', '.')
    text = text.replace('.', '. ')
    text = text.replace('!', '! ')
    text = text.replace('?', '? ')
    return text

@st.cache(suppress_st_warning=True)
def load_model():
    model = GPT2LMHeadModel.from_pretrained('last_1')
    model.cuda()
    return model

@st.cache(suppress_st_warning=True)
def generate_sequences(
        model, 
        prefix,
        num_sequences,
        num_sentences,
        do_sample,
        num_beams,
        length_penalty,
        top_p,
        temperature,
    ):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.cuda()

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1000,
        do_sample=do_sample,
        num_beams=num_beams,
        length_penalty=length_penalty,
        top_p=top_p,
        top_k=0,
        temperature=temperature,
        num_return_sequences=num_sequences,
    )
    
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []
    for seq in output_sequences:
        seq = seq.tolist()
        text = tokenizer.decode(seq, clean_up_tokenization_spaces=True)
        text = text[len(prefix):]
        text = clear_output_text(text)
        text = "".join(tokenize.sent_tokenize(text)[:num_sentences])
        generated_sequences.append(text)
    return generated_sequences