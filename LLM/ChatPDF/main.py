import os
from pathlib import Path
from utils import load_pdfs, create_vector_store, setup_llm_and_prompt, create_qa_chain, chat_with_pdf


def main():
    """
    Main function to run the PDF chat application.
    """
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Please set the GROQ_API_KEY environment variable")

    # List of PDF paths
    pdf_folder_path = Path('pdfs/')
    pdf_paths = [str(f) for f in pdf_folder_path.iterdir() if f.is_file() and f.suffix == '.pdf']

    try:
        # Load and process PDFs
        document_chunks = load_pdfs(pdf_paths)

        # Create vector store
        vectorstore = create_vector_store(document_chunks)

        # Setup LLM and prompt
        llm, qa_prompt = setup_llm_and_prompt()

        # Create QA chain
        qa_chain = create_qa_chain(llm, vectorstore, qa_prompt)

        # Start chat interface
        chat_with_pdf(qa_chain)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()