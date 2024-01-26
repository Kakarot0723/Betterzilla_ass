
import gradio as gr
import logging
import faiss       
import PyPDF2
import pdfreader
import openai
from InstructorEmbedding import INSTRUCTOR
from langchain.document_loaders import DirectoryLoader, TextLoader
logging.getLogger().setLevel(logging.CRITICAL)

num_words = 150
k = 7

messages = [
 {"role": "system", "content" : "You are an AI agent that summarizes chat in less than three setences."}
]

chats = [{"role": "system", "content" : "You are an AI assistant providing helpful advice. \n" + \
          "You are given the following extracted parts of a long document and a question. \n" + \
          "Provide a conversational answer based on the context provided. \n" + \
          "Give answer less than 500 words. \n" + \
          "Give answer in 4 sentences or less. \n" + \
          "SUMMARISE THE ANSWER IN 2 SENTENCES OR LESS. \n" + \
          "If you can't find the answer in the context below, use your prior knowledge,  \n" + \
          "but in most of the cases the answer will be in the context.' \n" + \
        #   "If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. \n" + \
          "Answer in Markdown format. \n" }]


model = INSTRUCTOR('hkunlp/instructor-large')
instruction = "Represent the query for retrieval:"

global index
sentences = None
index = faiss.IndexFlatL2(768)

def extract_text(pdf_file):
    
    with open (pdf_file.name, "rb") as f:
        pdf_reader = pdfreader.PdfFileReader(f)
        text = ""
        for page in range(pdf_reader.getNumPages()):
            page_obj = pdf_reader.getPage(page)
            text += page_obj.extractText()
        return text
    
def build_the_bot(openai_key):

    openai.api_key = openai_key
    print("openai key:", openai_key)
    global sentences
    with open('data.txt','r') as file:
        input_text = " ".join(line.rstrip() for line in file)
    #print(text)
    # input_text = TextLoader('data.txt')
    # print(input_text)
    print("text length:", len(input_text))
    words = input_text.split(' ')
    sentences = [words[i:i+num_words] for i in range(0, len(words), num_words)]
    sentences = [" ".join(sentences[i]) for i in range(len(sentences))] 

    print("number of sentences:", len(sentences))

    print("building the index")
    embeddings = model.encode([[instruction,i] for i in sentences])
    index.add(embeddings)              
    print(index.ntotal)

    return(input_text)

def chat(chat_history, user_input):
    
    global sentences
    messages.append({"role": "user", "content":"You are an AI agent that summarizes chat in less than 2 sentences and 500 words"+ "Question \n " + user_input})

    print(messages)

    print("Summarizing the chat history...")

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=messages
    )

    summarize = completion.choices[0].message.content
    print(f'Summarized Histoy: {summarize}')

    summarize = summarize + " \n question " + user_input
    print("Retrieving extra information...")
    xq = model.encode([[instruction,summarize]])
    D, I = index.search(xq, k)
    extra_info = ""
    for i in I[0]:
        try:
            extra_info += sentences[i] + " "
        except:
            print(len(sentences), i)

    chats.append({"role": "user", "content": "extra information = " + extra_info[:1000] + " \n question " + summarize[:500]})

    print("chats:", chats)
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=chats
    )

    chat_output = completion.choices[0].message.content
    print(f'ChatGPT: {chat_output}')

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": chat_output})
    yield chat_history + [(user_input, chat_output)]

with gr.Blocks() as demo:
    gr.Markdown('Chat with a PDF document')
    '''with gr.Tab("Select PDF"):
        #pdf = gr.File()
        openai_key = gr.Textbox(label="OpenAI API Key",)
        #text_output = gr.Textbox(label="PDF content")
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(build_the_bot, [openai_key])'''
    build_the_bot('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')
    with gr.Tab("Knowledge Bot"):
          chatbot = gr.Chatbot()
          message = gr.Textbox ("What is this document about?")
          message.submit(chat, [chatbot, message], chatbot)


demo.queue().launch(debug = True,share=True)