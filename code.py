!pip install langchain
!pip install openai
!pip install gradio
!pip install huggingface_hub

import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY="Key value"  #  Get this key, Go to https://platform.openai.com/account/api-keys; Create a new Secret Key; Copy the Secret Key for your use.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

template = """As an adventurous and globetrotting college student, you're constantly on the lookout for new cultures, experiences, and breathtaking landscapes. You've visited numerous countries, immersing yourself in local traditions, and you're always eager to swap travel stories and offer tips on exciting destinations.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

def get_text_response(user_message,history):
    response = llm_chain.predict(user_message = user_message)
    return response

demo = gr.ChatInterface(get_text_response, examples=["How are you doing?","What are your interests?","Which places do you like to visit?"])

if __name__ == "__main__":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.
