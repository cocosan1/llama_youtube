import streamlit as st
import numpy as np
import os
import textwrap # テキスト分割


from llama_hub.youtube_transcript import YoutubeTranscriptReader

from llama_index import (
    GPTVectorStoreIndex, # index化
    StorageContext, # index保存
    ServiceContext, # サービス全般の設定
)

from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.faiss import FaissReader #queryに類似したテキストのみを使ってindexを作成
from llama_index.callbacks import CallbackManager, LlamaDebugHandler # 処理のロギング
from llama_index.prompts.prompts import QuestionAnswerPrompt #コンテキストに対して回答をもとめるようなプロンプト

import faiss #ベクター検索ライブラリ。意味が近い文書を検索

# pip install youtube_transcript_api
# pip install faiss-cpu
# pip install nltk

st.set_page_config(page_title='テキスト抽出app')
st.title('テキスト抽出app from YouTube')
# 環境変数設定
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# 言語選択
selct_language = st.selectbox('言語を選択 ja: 日本語 en: 英語', ['ja', 'en'], key='lang')

#文量の最大値設定
if selct_language == 'ja':
    MAX_WORD_LENGTH = 2000
elif selct_language == 'en':
    MAX_WORD_LENGTH = 5000

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
    # テキストの文量がMAX_CHUNK_LENGTHを超える場合は分割 list
    chunks = textwrap.wrap(doc_text, width=MAX_WORD_LENGTH, break_long_words=False)
    len_chunks = len(chunks)

    st.write(f'words: {len(doc_text)} / len_chunks: {len_chunks}')

    #リンク
    st.write("chatgpt [link](https://chat.openai.com/)")


    return chunks, len_chunks

############ テキストデータの読み込み・index化
def make_index(documents):
 
    # Indexの作成　
    # 各Nodeに対応する埋め込みベクトルと共に順序付けせずに保持
    # 埋め込みベクトルを使用してNodeを抽出し、それぞれの出力を合成
    index = GPTVectorStoreIndex.from_documents(documents)

    # persistでstorage_contextを保存
    # Vector Storeはベクトルデータを格納/Document Storeはテキストデータ/Index Storeはインデックスに関する情報
    index.storage_context.persist(persist_dir="./storage_context")

    st.caption('index化完了')

###################################################################メインの関数
def non_select():
    st.info('項目を選択してください')
    st.stop()

def get_alltext():
    #テキストの抽出
    doc_text = get_text()

    #テキストの分割
    chunks, len_chunks = divide_text(doc_text)

    #チャンク数に合わせて数字のリストの作成
    chunk_nums = [i for i in range(len_chunks)]

    #日本語　chunksにindex指定しテキストの抽出
    if selct_language == 'ja':  
        for num in chunk_nums:
            st.write(chunks[num])
    
    #英語: chunksにindex指定しテキストの抽出　プロンプト作成
    elif selct_language == 'en':
        for num in chunk_nums:
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
                あなたはプロの編集者です。\
                下記の制約条件に従って提示文を要約してください。\
                \
                # 制約条件\
                - 登場人物の考えや発言、ふるまいに関する情報は削らずに盛り込んでください。\
                - 具体的な情報については削らずに盛り込んでください。\
                - {amount_text}の文量になるようしてください。\
                - 必ず日本語で表示してください。\
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
                あなたはプロの翻訳者兼編集者です。\
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
                3 必ず日本語に訳して表示する。\
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


    # インスタンス化
    loader = YoutubeTranscriptReader()

    #youtubeからリストデータの抽出
    documents = loader.load_data(ytlinks=[url], languages=[selct_language])

    #index化
    make_index(documents)

    #ストレージからindexデータの読み込み
    def read_storage():
        #from_defaults indexを保存、読み込みする時の設定をデフォルト値で設定
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
        )
        return storage_context
    
    storage_context = read_storage()

    # embeddingモデルのインスタンス化
    embed_model = OpenAIEmbedding()

    # 準備
    docs = [] # 埋め込みベクトルを保持するためのリスト
    id_to_text_map = {} # 文書のIDとテキストの対応を保持するための辞書
    
    #文書データを格納しているstorage_contextから文書の一覧を取得
    for i, (_, node) in enumerate(storage_context.docstore.docs.items()):
        #文書ノード（node）からテキストを取得
        text = node.get_text()
        #テキストの埋め込みベクトルを生成、リストに保存
        docs.append(embed_model.get_text_embedding(text))
        id_to_text_map[i] = text
    # listをnp配列化
    docs = np.array(docs)

    #text-ada-embedding-002から出力されるベクトル長を指定
    d = 1536
    # faiss 高次元ベクトルを高速に検索
    # IndexFlatL2 faissの中で最も基本的な索引（index）の種類。
    # 索引とは、ベクトルの集合を管理し、検索を効率的に行うためのデータ構造
    # L2距離（ユークリッド距離）を使って、全てのベクトルとの距離を計算して、最も近いベクトルを返す
    # 精度は高いが、メモリや計算時間が多く必要

    #インスタンス化
    index = faiss.IndexFlatL2(d)
    #Faissにベクトルを登録
    index.add(docs)

    # questionのベクトル化
    query = embed_model.get_text_embedding(question)
    # queryをnp配列化
    query=np.array([query])

    # FaissReader クエリに類似したテキストのみを使ってインデックスを作成
    # インスタンス化
    reader = FaissReader(index)
    documents = reader.load_data(query=query, id_to_text_map=id_to_text_map, k=num_node)

    st.write(f'count_node: {len(documents)}')

    ######### デバッグ用
    llama_debug_handler = LlamaDebugHandler() # 処理のロギング
    # 各処理フェーズにおけるstart, endにおけるコールバックをhandlerとして設定
    callback_manager = CallbackManager([llama_debug_handler]) 

    # インデックスを作成したりクエリを実行する際に必要になる部品（LLMの予測器、埋め込みモデル、プロンプトヘルパーなど）
    # をまとめたオブジェクトを作成するメソッド
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    #Faissで抽出した類似したノードを使って、GPTVectorStoreIndexを作成。
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    if selct_language == 'ja':
        #質問用のQAプロンプトを生成
        QA_PROMPT_TMPL = (
            "私たちは以下の情報をコンテキスト情報として与えます。 \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "あなたはAIとして、この情報をもとに質問に対して必ず日本語で答えます。: {query_str}\n"
        )
        # QuestionAnswerPrompt 質問と回答の形式で外部データにアクセスできる
        # コンテキストに対して回答をもとめるようなプロンプト形式
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
    
    #LlamaIndex のプロンプトをカスタマイズする
    query_engine = index.as_query_engine(text_qa_template=qa_prompt)

    # questionを送る
    response = query_engine.query(question)

    #responseからtextの取り出し。sourceも取り出し可。
    response_text = response.response.replace("\n", "")

    # チャット画面に表示
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


        
        