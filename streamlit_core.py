import streamlit as st
import numpy as np
import torch

from generation import generate_sequences, load_model
from youtube_parser import get_info_from_url

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@st.cache(suppress_st_warning=True)
def get_video_info():
    url = st.text_area('Insert YouTube video URL')
    if url:
        url = url.replace("/n", "")
        video_info = get_info_from_url(url)
        prefix = ". ".join([video_info["Channel"], video_info["Title"]])
        st.write("URL successfully parsed")
    return prefix

def print_results(sequences):
    with st.container():
        st.header("Generated comments")
        for i, text in enumerate(sequences):
            st.subheader(f"Comment №{i+1}")
            st.write(text)
    st.header("Thank you for your participation!")
    st.balloons()

def main():
    model = load_model()

    st.title("Create your YouTube comments")
    st.caption("The maximum comment length is 1000 words")

    url = st.text_area('Insert YouTube video URL')
    if url:
        url = url.replace("/n", "")
        video_info = get_info_from_url(url)
        prefix = ". ".join([video_info["Channel"], video_info["Title"]]) + ". "
        st.caption("URL successfully parsed")
        st.caption(f"Prefix: {prefix}")

        num_sequences = None
        num_sentences = None
        do_sample = False
        num_beams = None
        top_p = None
        temperature = None
        length_penalty = None

        with st.container():
            st.header("Set comments parameters")
            num_sequences = st.slider('Number of comments: ', 1, 10)
            num_sentences = st.slider('Number of sentences in each comment: ', 1, 10)

            st.header("Set GPT-2 decode parameters")
            decode_method = st.selectbox(
                "Select decoding method",
                ('None', 'Top-p sampling', 'Beam search')
            )

            if decode_method != 'None':
                if decode_method == 'Beam search':
                    num_beams = st.slider('Number beams: ', 0, 50, 5)
                    length_penalty = st.slider('Length penalty: ', 0., 1., 0.8)

                if decode_method == 'Top-p sampling':
                    do_sample = True
                    choice_settigns = st.radio(
                        "Select settings",
                        ('Top-p', 'Temperature')
                    )
                    if choice_settigns == 'Temperature':
                        temperature = st.slider('Temperature: ', 0., 1., 0.6)
                    else:
                        top_p = st.slider('Top_p: ', 0., 1., 0.5)

                if st.button('Run comments generation'):
                    sequences = generate_sequences(
                        model,
                        prefix,
                        num_sequences,
                        num_sentences,
                        do_sample,
                        num_beams,
                        length_penalty,
                        top_p,
                        temperature,
                    )
                    print_results(sequences)

if __name__ == "__main__":
    main()