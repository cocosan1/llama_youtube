import streamlit as st
import numpy as np
import os
import datetime
import textwrap # テキスト分割
import shutil # ディレクトリ削除

from llama_hub.youtube_transcript import YoutubeTranscriptReader

from llama_index import (
    GPTVectorStoreIndex,
    StorageContext,
    ServiceContext,
)

from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.faiss import FaissReader
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.prompts.prompts import QuestionAnswerPrompt

import faiss #ベクター検索ライブラリ。意味が近い文書を検索

st.title('テキスト抽出app from YouTube')
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# 言語選択
selct_language = st.selectbox('言語を選択 ja: 日本語 en: 英語', ['ja', 'en'], key='lang')

#文量の最大値設定
if selct_language == 'ja':
    MAX_CHUNK_LENGTH = 1500
elif selct_language == 'en':
    MAX_CHUNK_LENGTH = 3800

# url入力
url = st.text_input('youtubeのURLを入力', key='url')

if url == '':
    st.info('youtubeのURLを入力してください')
    st.stop()

########### youtubeからテキストを取得
def get_text():

    # インスタンス化
    loader = YoutubeTranscriptReader()

    #youtubeからリストデータの抽出
    documents = loader.load_data(ytlinks=[url], languages=[selct_language])

    #リストからxmlデータの抽出
    doc = documents[0]
    #xmlデータからテキストの抽出
    doc_text = doc.text

    return doc_text

########### テキストを最大値に合わせて分割
def divide_text(doc_text):
    # テキストの文量がMAX_CHUNK_LENGTHを超える場合は分割
    chunks = textwrap.wrap(doc_text, width=MAX_CHUNK_LENGTH, break_long_words=False)
    len_chunks = len(chunks)

    st.write(f'words: {len(doc_text)} / len_chunks: {len_chunks}')

    #リンク
    st.write("chatgpt [link](https://chat.openai.com/)")


    return chunks, len_chunks

############ indexを作成
def make_index(documents):
 
    #テキストデータの読み込み・index化
    # Indexの作成
    index = GPTVectorStoreIndex.from_documents(documents)
    # persistでstorage_contextを保存
    index.storage_context.persist(persist_dir="./storage_context")

    st.caption('index化完了')

def non_select():
    st.info('項目を選択してください')
    st.stop()

def get_alltext():
    #テキストの抽出
    doc_text = get_text()

    #テキストの分割
    chunks, len_chunks = divide_text(doc_text)

    #チャンク数に合わせて数字のリストの作成
    chunks_nums = [i for i in range(len_chunks)]

    #日本語　chunksにindex指定しテキストの抽出
    if selct_language == 'ja':  
        for num in chunks_nums:
            st.write(chunks[num])
    
    #英語: chunksにindex指定しテキストの抽出　プロンプト作成
    if selct_language == 'en':
        for num in chunks_nums:
            st.write(f'prompt: {num}-------------------------------------------------')
            chunk = f'\
                あなたはプロの翻訳者です。\
                下記の英文を和訳してください。\
                # 英文\
                {chunks[num]}'

            st.code(chunk)
    
def make_summary():
    #文量の指定
    amount_texts = ['75%程度', '50%程度', '30%程度']
    amount_text = st.selectbox('文量を指定', amount_texts, key='at')
    #テキストの抽出
    doc_text = get_text()

    #テキストの分割
    chunks, len_chunks = divide_text(doc_text)
    #チャンク数に合わせて数字のリストの作成
    chunks_nums = [i for i in range(len_chunks)]

    #日本語　chunksにindex指定しテキストの抽出
    if selct_language == 'ja':  
        for num in chunks_nums:
            st.write(f'prompt: {num}-------------------------------------------------')
            prompt = f'\
                # 指示書\
                あなたはプロの雑誌編集者です。\
                下記の制約条件に従って提示文を要約してください。\
                \
                # 制約条件\
                - 登場人物の考えや発言、ふるまいに関する情報は削らずに盛り込んでください。\
                - 具体的な情報については削らずに盛り込んでください。\
                - {amount_text}の文量になるようしてください。\
                \
                # 提示文\
                {chunks[num]}'

            st.code(prompt)
    
    #英語: chunksにindex指定しテキストの抽出　プロンプト作成
    if selct_language == 'en':
        for num in chunks_nums:
            st.write(f'prompt: {num}-------------------------------------------------')
            prompt = f'\
                # 指示書\
                あなたはプロの翻訳者兼雑誌編集者です。\
                下記の制約条件と手順に従って提示文を要約してください。\
                \
                # 制約条件\
                - 登場人物の考えや発言、ふるまいに関する情報は削らずに盛り込んでください。\
                - 具体的な情報については削らずに盛り込んでください。\
                - {amount_text}の文量になるようしてください。\
                \
                # 手順\
                1 英語に翻訳する。\
                2 制約条件に従い要約する。\
                3 日本語に訳して表示する。\
                \
                # 提示文\
                {chunks[num]}'

            st.code(prompt)
def q_and_a():
     #質問の入力
    question = st.text_input('質問を入力してください', key='question')
    num_node = st.number_input('ノード数指定', value=1, key='num_node')

    if not question:
        st.info('質問を入力してください')
        st.stop()
    
    #storage_contextの初期化
    file_path = "./storage_context"

    if os.path.exists(file_path):
        #指定したファイルパス（またはディレクトリパス）が実際に存在するかどうかを確認
        shutil.rmtree(file_path)
        st.markdown('#### process')
        st.caption("storage_contextフォルダを削除しました。")
        st.caption('テキストのindex化作業を開始します。')
    else:
        st.markdown('#### process')
        st.caption("storage_contextフォルダ は存在しません。")
        st.caption('テキストのindex化作業を開始します。')

    # インスタンス化
    loader = YoutubeTranscriptReader()

    #youtubeからリストデータの抽出
    documents = loader.load_data(ytlinks=[url], languages=[selct_language])

    #index化
    make_index(documents)

    #ストレージからindexデータの読み込み
    def read_storage():
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
        )
        return storage_context
    
    storage_context = read_storage()

    # embeddingモデルのインスタンス化
    embed_model = OpenAIEmbedding()

    # 埋め込みベクトルを保持するためのリスト
    docs = []

    # 文書のIDとテキストの対応を保持するための辞書
    id_to_text_map = {}

    #文書データを格納しているストレージコンテキストから文書の一覧を取得
    for i, (_, node) in enumerate(storage_context.docstore.docs.items()):
        #文書ノード（node）からテキストを取得
        text = node.get_text()
        #テキストの埋め込みを生成します
        docs.append(embed_model.get_text_embedding(text))
        id_to_text_map[i] = text
    docs = np.array(docs)

    #text-ada-embedding-002から出力されるベクトル長を指定
    d = 1536
    index = faiss.IndexFlatL2(d)
    #Faissにベクトルを登録
    index.add(docs)

    # クエリとFaissから取り出すノード数の設定
    query_text = question

    # questionのベクトル化
    query = embed_model.get_text_embedding(query_text)
    query=np.array([query])

    # Faissからのquestionに近いノードの取り出し
    reader = FaissReader(index)
    documents = reader.load_data(query=query, id_to_text_map=id_to_text_map, k=num_node)

    st.write(f'count_node: {len(documents)}')

    # デバッグ用
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    #Faissで確認した類似したノードを使って、GPTListIndexを作成。
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    if selct_language == 'ja':
        #質問用のQAプロンプトを生成
        QA_PROMPT_TMPL = (
            "私たちは以下の情報をコンテキスト情報として与えます。 \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "あなたはAIとして、この情報をもとに質問を日本語で答えます。: {query_str}\n"
        )
        qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    
    if selct_language == 'en':
        #質問用のQAプロンプトを生成
        QA_PROMPT_TMPL = (
            "We provide the following information as contextual information. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "You, as an AI, will answer questions in Japanese based on this information.: {query_str}\n"
        )
        qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    
    query_engine = index.as_query_engine(text_qa_template=qa_prompt)

    # テンプレを送る
    response = query_engine.query(question)

    #responseからtextとsourseの取り出し
    response_text = response.response.replace("\n", "")

    with st.chat_message("user"):
        st.write(question)
    
    message = st.chat_message("assistant")
    message.write(response_text)




def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '--': non_select,
        '全文抽出': get_alltext,
        '要約': make_summary,
        'Q&A': q_and_a

    }
    selected_app_name = st.selectbox(label='項目の選択',
                                                options=list(apps.keys()))


    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

if __name__ == '__main__':
    main()


        
        