import os
import textwrap
from pathlib import Path  
from dotenv import load_dotenv
from llama_parse import LlamaParse  

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")


def print_response(response) :
    response_text = response["result"]
    for chunk in response_text.split("\n") :
        if not chunk :
            print()
            continue
        print("\n".join(
            textwrap.wrap(chunk, 100, break_long_words=False)
            )
        )


instruction_for_parser = """The provided document is a research paper that was submitted by Google in the year of 2017. 
The paper deals with the idea for a new neural network architecture coined as Transformers, an encoder-decoder model that can be used for machine translation tasks. 
The paper includes many significant tables and diagrams that convey important meanings and descriptions. 
In the document many mathematical equations are also present that convey important meanings and explanations for important concepts. 
Try to be precise while answering the questions.  
"""

print("[SYS INFO] ----> Document Parsing Is Ongoing.....{}".format("\n"))
parser = LlamaParse(
    api_key = LLAMA_PARSE_API_KEY, 
    result_type = "markdown", 
    parsing_instruction = instruction_for_parser, 
    max_timeout = 5000,
)

llama_parse_documents = parser.load_data("Documents\\attention-is-all-you-need.pdf")
parsed_doc = llama_parse_documents[0]

document_path = Path("Documents\\parsed_document.md")
with document_path.open("a", encoding="utf-8") as fp :
    fp.write(parsed_doc.text)

print("[SYS INFO] ----> Parsed Document Loaded and Saved Successfully....{}".format("\n"))

