import pandas as pd
from datasets import load_dataset
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ast import literal_eval

def generate_eval_question(problem_statement, chain):
    """
    Generate a technical question based on the problem statement.
    
    Args:
        problem_statement (str): The problem statement to base the question on
        prompt (PromptTemplate): The prompt template for question generation
        
    Returns:
        str: Generated question
    """
    try:
        response = chain.invoke({"problem_statement": problem_statement})
        print(f"Generated response: {response}")
        return response.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def main():
    sklearn_df = pd.read_csv("./graph/sklearn/eval_df.csv")
    print(sklearn_df.head())
    def reformat_function_names(edit_functions):
        return [func_name.split(":")[1] if ":" in func_name else func_name for func_name in edit_functions]
    
    sklearn_df["edit_functions"] = sklearn_df["edit_functions"].apply(literal_eval)
    sklearn_df["edit_functions"] = sklearn_df["edit_functions"].apply(reformat_function_names)
    print(f"Number of sklearn questions: {sklearn_df.shape}")

    question_generation_template = """
You are helping evaluate a codebase question answering system. Based on the following problem statement from a GitHub issue, generate a single, clear, technical query that could be answered by reading the relevant code or documentation.

### Problem Statement:
{problem_statement}

### Instructions:
- Generate a **precise and technical** query/question.
- The query should be at most 20 words.
- Do **not** repeat the same phrase multiple times.
- The output should be a **single query**, starting with an uppercase letter and ending with a question mark, or exclamation point.

### Output:
Question:"""

    question_prompt = PromptTemplate.from_template(question_generation_template)


    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=text_generator)

    
        
    generation_chain = question_prompt | llm
    
    print("chain created")

    sklearn_df["question"] = None
    for idx,row in sklearn_df.iterrows():
        problem_statement = row["problem_statement"]
        question = generate_eval_question(problem_statement, generation_chain)
        sklearn_df.loc[idx, "question"] = question

    # Save the updated DataFrame to a CSV file
    sklearn_df.to_csv("./graph/sklearn/eval_df.csv", index=False)
        
if __name__ == "__main__":
    main()