import sys
import nbformat
from nbclient import NotebookClient
import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def run_notebook(question):
    notebook_path = "chatbot_example/chatbot.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    notebook.cells.insert(0, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": f"question = '{question}'"
    })

    notebook.cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": f"get_chatbot_response('{question}')"
    })

    # Converter o notebook para cns ler
    notebook = nbformat.from_dict(notebook)
    client = NotebookClient(notebook, kernel_name="python")
    client.execute()

    # Extrair a resposta da ultima celula
    last_cell = notebook.cells[-1]  # Pega na última célula
    if last_cell.cell_type == "code" and last_cell.outputs:
        for output in last_cell.outputs:
            if output.output_type in ["stream", "execute_result"]:
                if "data" in output:
                    if "text/plain" in output["data"]:
                        response = output["data"]["text/plain"].strip("'").replace("\\n", "\n").replace("\\t", "\t")
                        return response

    return "Erro ao processar a resposta da última célula."


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Nenhuma pergunta fornecida."
    print(str(run_notebook(question)))
