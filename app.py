import streamlit as st
import os
import textwrap

from llama_hub.youtube_transcript import YoutubeTranscriptReader

st.markdown('### llamaindex youtube')
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

MAX_CHUNK_LENGTH = 1500


url = st.text_input('url of youtube')

if url == '':
    st.info('input url of youtube')
    st.stop()

else:

    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[url], languages=['ja'])

    #リスト内の要素を抽出
    doc = documents[0]
    #textを抽出（xml形式）
    doc_text = doc.text

    st.write(f'文字数: {len(doc_text)}')

    # テキストの長さが長すぎる場合、その位置で分割
    chunks = textwrap.wrap(doc_text, width=MAX_CHUNK_LENGTH, break_long_words=True)

    #リンク
    st.write("chatgpt [link](https://chat.openai.com/)")

    len_chunks = len(chunks)
    chunks_nums = [i for i in range(len_chunks)]
    
    slct_num = st.selectbox('chunk_noを選択', chunks_nums, key='slct_num')

    chunk = f'あなたは優秀な編集者です。下記の文章を1/3程度の文量で要約してください。{chunks[slct_num]}'

    st.code(chunk)


        
        