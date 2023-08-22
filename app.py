import streamlit as st
import os

from llama_hub.youtube_transcript import YoutubeTranscriptReader

st.markdown('### llamaindex youtube')
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

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

    prompt_txt = f'あなたは優秀な編集者です。下記の文章を1/3程度の文量で要約してください。{doc_text}'

    #リンク
    st.write("chatgpt [link](https://chat.openai.com/)")

    st.code(prompt_txt)