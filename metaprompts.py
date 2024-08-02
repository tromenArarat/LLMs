import os
from langchain.llms import Cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

llm = Cohere()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente en el desarrollo de un proyecto de: "+
            "{tema}"+
            "responde con la mayor erudición posible, "+
            "cita las fuentes siempre que te sea posible",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | llm

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

input = "cómo puedo convertir un pdf a texto con herramientas"
+"parecidas a las que provee BeatifulSoap respecto a html y xml"

result = with_message_history.invoke(
    {"tema": "customización de un modelo de llm", 
     "input": input },
    config={"configurable": {"session_id": "abc123"}},
)

print(result)

##  ¡Hola amigo! 

# El método mas sencillo para convertir un PDF a texto es usar un editor de código en línea como la plataforma [CodeSandbox](https://codesandbox.io/):

# 1. Arranca un nuevo proyecto en el cual introduce el HTML y cuando te des la opcion de elegir el tipo de contenido desactive todo excepto HTML y CSS.
# 2. En el codigo agrega una segunda etiqueta `div` en la cual Cree un objeto para el plug-in de PDF.js que se puede instalar desde la consola.

# Esto sería el código para la etiqueta `div` con un nombre de tu elección:

# ```html
# <div id="pdf-container"></div>
# ```

# 3. Ahora busca en [GitHub](https://github.com/) el repositorio de PDF.js y sigue las instrucciones de [PDF.js Web Assembly](https://github.com/mozilla
