import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 1. 定义常量 ---
MODEL_PATH = "./models/qwen1_5-1_8b-chat-q4_k_m.gguf"
KNOWLEDGE_DIR = "./knowledge"
DB_PATH = "./chroma_db"
EMBED_MODEL = "local:BAAI/bge-small-zh-v1.5"

def main():
    # --- 2. 配置 LlamaIndex 的全局设置 ---
    print("正在配置模型...")
    
    Settings.llm = LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_new_tokens=512,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 0},
        messages_to_prompt=lambda messages: (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{messages[0].content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        completion_to_prompt=lambda completion: (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{completion}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        verbose=False,
    )
    Settings.embed_model = resolve_embed_model(EMBED_MODEL)

    # --- 3. 加载或创建向量数据库 ---
    print("正在加载/创建知识库...")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("my_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() == 0:
        print("知识库为空，正在从文件创建...")
        documents = SimpleDirectoryReader(KNOWLEDGE_DIR).load_data()
        # 确保有文件被加载
        if not documents:
            print("错误：'knowledge' 文件夹为空或无法读取文件。请添加 txt 文件。")
            return # 退出程序
            
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(f"知识库创建成功！已添加 {chroma_collection.count()} 个知识片段。")
    else:
        print(f"成功加载本地知识库，其中包含 {chroma_collection.count()} 个知识片段。")
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
    # --- 4. 创建查询引擎 ---
    print("正在创建查询引擎...")
    qa_prompt_tmpl_str = (
        "已知信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据以上已知信息，回答以下问题。不要使用任何你自己的知识。\n"
        "问题: {query_str}\n"
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    query_engine = index.as_query_engine(
        text_qa_template=qa_prompt_tmpl,
        similarity_top_k=2
    )

    # --- 5. 启动交互式问答循环 ---
    print("\n✅ 个人AI助手已就绪，开始提问吧！(输入 'exit' 退出)")
    while True:
        query = input("你: ")
        if query.lower() == 'exit':
            break
        
        print("--- 步骤1: 正在从知识库检索相关信息... ---")
        retriever = index.as_retriever(similarity_top_k=2)
        retrieved_nodes = retriever.retrieve(query)
        
        if retrieved_nodes:
            print("--- 检索成功！找到以下内容作为上下文：---")
            for node in retrieved_nodes:
                print(f"【内容片段】: {node.get_text().strip()}...")
                print("--------------------")
        else:
            print("--- 检索失败：未在你的知识库中找到任何相关内容。---")

        print("\n--- 步骤2: AI 正在基于以上信息思考... ---")
        response = query_engine.query(query)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
