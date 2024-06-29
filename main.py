# based cohere agent doc.
import cohere
import os
import pandas as pd
import json
import PyPDF2
#https://dashboard.cohere.com/api-keys
co = cohere.Client(api_key="<YOU-COHERE-KEY>")

COHERE_MODEL = 'command-r-plus'

def read_then_extract_pdf(file:str) -> dict :
  
  pdf_file = open(file, 'rb')

  #read pdf file
  reader = PyPDF2.PdfReader(pdf_file)

  # get first page.
  first_page = reader.pages[0]

  # extract text
  text = first_page.extract_text()


  print(text)

  pdf_file.close()

  return {"result": text}


def convert_to_json(text: str) -> dict:
    """
    Given text files, convert to json object and saves to csv.

    Args:
        text (str): The text to extract information from.

    Returns:
        dict: A dictionary containing the result of the conversion process.
    """

    MANDATORY_FIELDS = [
        "total_amount",
        "invoice_number",
    ]

    message = """# Instruction
    Given the text, convert to json object with the following keys:
    total_amount, invoice_number

    # Output format json:
    {{
        "total_amount": "<extracted invoice total amount>",
        "invoice_number": "<extracted invoice number>",
    }}

    Do not output code blocks.

    # Extracted PDF
    {text}
    """

    result = co.chat(
        message=message.format(text=text), model=COHERE_MODEL, preamble=None
    ).text

    try:
        result = json.loads(result)
        # check if all keys are present
        if not all(i in result.keys() for i in MANDATORY_FIELDS):
            return {"result": f"ERROR: Keys are missing. Please check your result {result}"}

        df = pd.DataFrame(result, index=[0])
        df.to_csv("output.csv", index=False)
        return {"result": "SUCCESS. All steps have been completed."}

    except Exception as e:
        return {"result": f"ERROR: Could not load the result as json. Please check the result: {result} and ERROR: {e}"}


pdf_path = "./simple_invoice.pdf"
prompt = f"""
    # Instruction
    You are expert at extracting invoices from PDF. The text of the PDF file is given below.

    You must follow the steps below:
    1. Read the text from the PDF file path.
    2. Summarize the text and extract only the most information: total amount billed and invoice number.
    3. Using the summary above, call convert_to_json tool, which uses the summary from step 2.
    If you run into issues. Identifiy the issue and retry.
    You are not done unless you see SUCCESS in the tool output.

    # Pdf File Path:
    {pdf_path}
    """

functions_map = {
        "convert_to_json": convert_to_json,
        "read_then_extract_pdf": read_then_extract_pdf,
}

tools = [
    {
        "name": "convert_to_json",
        "description": "Given a text, convert it to json object.",
        "parameter_definitions": {
            "text": {
                "description": "text to be converted into json",
                "type": "str",
                "required": True,
            },
        },
    },
    {
        "name": "read_then_extract_pdf",
        "description": "Given a pdf file, read then extract pdf file to str text.",
        "parameter_definitions": {
            "file": {
                "description": "pdf file path ",
                "type": "str",
                "required": True,
            },
        },
    },
]


response = co.chat(
    model=COHERE_MODEL,
    message=prompt,
    preamble=None,
    tools=tools,
)
while response.tool_calls:

  tool_results = []

  for tool_call in response.tool_calls:
      print("tool_call.parameters:", tool_call.parameters)
      if tool_call.parameters:
          output = functions_map[tool_call.name](**tool_call.parameters)
      else:
          output = functions_map[tool_call.name]()
      print("output:", output)
      outputs = [output]
      tool_results.append({"call": tool_call, "outputs": outputs})

      

      print(
          f"= running tool {tool_call.name}, with parameters: {tool_call.parameters}"
      )
      print(f"== tool results: {outputs}")
  
  response = co.chat(
            model=COHERE_MODEL,
            message="",
            chat_history=response.chat_history,
            preamble=None,
            tools=tools,
            tool_results=tool_results,
        )

