import os
from api_key import api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ['OPENAI_API_KEY'] = api_key

st.title('🦜️🔗 IITI Creator')
prompt = st.text_input('plug in your prompt here')

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title {topic}'
)

# llms
llm = OpenAI(temperature=0.9)

# using hugging face
# llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":1e-10})

title_chain =LLMChain(llm=llm,prompt=title_template,verbose=True)



if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)
