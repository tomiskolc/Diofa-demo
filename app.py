# packages
import streamlit as st
import numpy as np
import requests

#from credentials import openai_api
import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

#st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg", use_column_width=True)
with st.sidebar:
    openai_api = st.text_input('OpenAI API Key', type = 'password', key = 'openai_key')
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api

MODEL_RELEVANT_DOC_NUMBER = 3
MODEL_INPUT_TOKEN_SUMM_LIMIT = 3200
MODEL_MAX_TOKEN_LIMIT = 4097
MODEL_COST = 0.0015

model_id = "intfloat/multilingual-e5-large"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def inference_embedding(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()
@st.cache_resource
def load_embedding():
	  return HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = load_embedding()

# functions, prompts
def generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS):
    completion = openai.ChatCompletion.create(
        model=MODEL, 
        messages=messages, 
        temperature=TEMPERATURE, 
        max_tokens=MAX_TOKENS)
    return completion.choices[0]['message']['content']

def retrieve_relevant_chunks(user_input, db):

    query_embedded = inference_embedding(user_input)

    sim_docs = db.similarity_search_by_vector(query_embedded, k = MODEL_RELEVANT_DOC_NUMBER)
    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in sim_docs]
    sources = "\n".join(results)

    return sources


default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
If you did not use the information below to answer the question, do not include the source name or any square brackets."""

system_message = """{system_prompt}

Sources:
{sources}

"""

question_message = """
{question}

Assistant: 
"""


# streamlit app
st.title("Diófa - ChatGPT")
st.header("Proof-of-Concept for integrating Generative AI with Diófa documents")
st.write("Developed by Hiflylabs")

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    You may set the following settings\n

    1. Prompt parameters
        - System message
        - max_tokens
        - temperature"""
)

MODEL = st.radio('Select the OpenAI model you want to use', 
                 ['gpt-3.5-turbo', 'gpt-4'], horizontal=True)

prompt_expander = st.expander(label='Set your Prompt settings')
with prompt_expander:
    cols=st.columns(2)
    with cols[0]:
        SYSTEM_MESSAGE = st.text_area('Set a system message', value = default_system_prompt, height = 400)
    with cols[1]:
        TEMPERATURE = float(st.select_slider('Set your temperature', [str(round(i, 2)) for i in np.linspace(0.0, 2, 101)], value = '0.2')) 
        MAX_TOKENS = st.slider('Number of max output tokens', min_value = 1, max_value = MODEL_MAX_TOKEN_LIMIT-MODEL_INPUT_TOKEN_SUMM_LIMIT, value = 512)

#### LOAD INDEX ####

@st.cache_resource
def load_index():
	  return FAISS.load_local("faiss_index_e5_large", st.session_state['embeddings'])

if 'db' not in st.session_state:
    st.session_state['db'] = load_index()

#### END OF LOAD INDEX ####

if not openai_api:
    st.warning('🔑🔒 Paste your OpenAI API key on the sidebar 🔑🔒')
else:

    msg = st.chat_message('assistant')
    msg.write("Hello 👋 Ask me questions about Diófa's documents!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if QUERY := st.chat_input("Enter your question to Diófa here."):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(QUERY)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            relevant_chunks = retrieve_relevant_chunks(QUERY, st.session_state['db'])

            messages =[
                        {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
                        {"role": "user", "content": system_message.format(system_prompt = SYSTEM_MESSAGE, sources=relevant_chunks)},
                        {"role": "user", "content": question_message.format(question=QUERY)}
                        ]


            # display streaming response 
            report = []
            for resp in openai.ChatCompletion.create(model=MODEL, 
                                                    messages=messages,
                                                    max_tokens=MAX_TOKENS, 
                                                    temperature = TEMPERATURE,
                                                    stream = True):

                if resp.choices[0]['delta'] != {}:
                    report.append(resp.choices[0]['delta']['content'])
                else:
                    break
                result = "".join(report).strip()
                result = result.replace("\n", "")        
                message_placeholder.markdown(f'*{result}*') 

            st.session_state.messages.append({"role": "user", "content": QUERY})
            st.session_state.messages.append({"role": "assistant", "content": result})

        if len(st.session_state.messages) > 0:

            sources_expander = st.expander(label='Check sources identified as relevant')
            with sources_expander:
                #st.write('\n')
                #if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]:
                #    st.write('All sources were used within the prompt')
                #else:
                    #st.write("Below are the sources that have been identified as relevant:")

                st.text(relevant_chunks)
                # Így kell lehivatkozni, de vhogy ki kéne szedni a linkeket
                # st.write("[2022_kozzetetel/01_2022_02_VH_MPTIA sikerdíj_alairt.pdf](https://diofa.sharepoint.com/:b:/r/sites/Extranet/SUF/ChatGPT/2022_kozzetetel/01_2022_02_VH_MPTIA%20sikerd%C3%ADj_alairt.pdf?csf=1&web=1&e=1hDoeb)-page-1: 2022_kozzetetel/01_2022_02_VH_MPTIA sikerdíj_alairt --- 1  A DIÓFA ALAPKEZELŐ ZRT . VEZÉRIGAZGATÓJÁNAK")
