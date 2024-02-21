import ollama
import textwrap

class QueryLLM:
    def __init__(self, model="llama2"):
        self.model = model

    def wrap_text_preserve_newlines(self, text, width=110):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(self, llm_response):
        return(self.wrap_text_preserve_newlines(llm_response))

    def query(self, query):
        response = ollama.chat(model='llama2', messages=[
                {
                    'role': 'user',
                    'content': query,
                },
            ])
        return self.process_llm_response(response['message']['content'])
    

if __name__ == "__main__":
    query_llm = QueryLLM()
    response = query_llm.query("What is the capital of France?")
    print(response)