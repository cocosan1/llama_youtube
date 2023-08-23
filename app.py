import streamlit as st
import os
import textwrap

from llama_hub.youtube_transcript import YoutubeTranscriptReader

st.markdown('### llamaindex youtube')
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

doc_text = ''

with st.form('unput form', clear_on_submit=True):
    slct_language = st.selectbox('select language', ['en', 'ja'], key='language')

    amounttxts = ['全文掲載', '80%程度', '75%程度', '50%程度', '1/3程度']

    amounttxt = st.selectbox('select amount of writing', amounttxts, key='amounttxt')

    url = st.text_input('url of youtube')

    submitted = st.form_submit_button('submit')

if submitted:
    MAX_CHUNK_LENGTH = ''
    if slct_language == 'en':
        MAX_CHUNK_LENGTH = 3800

    elif slct_language == 'ja':
        MAX_CHUNK_LENGTH = 1500

    #########################################読み込み
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[url], languages=[slct_language])

    #リスト内の要素を抽出
    doc = documents[0]
    #textを抽出（xml形式）
    doc_text = doc.text

    # テキストの長さが長すぎる場合、その位置で分割
    chunks = textwrap.wrap(doc_text, width=MAX_CHUNK_LENGTH, break_long_words=True)
    len_chunks = len(chunks)

    st.write(f'words: {len(doc_text)} / len_chunks: {len_chunks}')

    #リンク
    st.write("chatgpt [link](https://chat.openai.com/)")

    
    chunks_nums = [i for i in range(len_chunks)]
    
    # slct_num = st.selectbox('chunk_noを選択', chunks_nums, key='slct_num')

    if amounttxt == '全文掲載':
        if slct_language == 'en':
            for num in chunks_nums:
                st.write(f'text: {num}-------------------------------------------------')
                chunk = f'\
                    あなたは優秀な翻訳者です。下記の提示文の日本語訳文を作成してください。\
                    #提示文\
                    {chunks[num]}'

                st.code(chunk)
        elif slct_language == 'ja':
            for num in chunks_nums:
                st.write(chunks[num])

    else:
        for num in chunks_nums:
            st.write(f'prompt: {num}-------------------------------------------------')
            chunk = f'\
                #命令書\
                あなたは優秀な記事の編集者です。\
                下記のルールと手順を守り、提示文の要約文を作成してください。\
                \
                #ルール\
                - 考えや気持ちが入った記述は取り残さないようにする。\
                - 具体的な事例が入ったは取り残さないようにする。\
                \
                #手順\
                1 英語の場合、日本語に訳する。\
                2 {amounttxt}の文量に収まるように要約する。\
                \
                #提示文\
                {chunks[num]}'

            st.code(chunk)


        
        