import pandas as pd
from ast import literal_eval
from qa_generator import QAPairGenerator

def main():
    df = pd.read_csv("./graph/sklearn/eval_df.csv")
    print(df.shape)
    for col in ["linked_prs", "linked_issues", "edit_functions", "labels"]:
        if col in df.columns:
            df[col] = df[col].apply(literal_eval)
            
    qa_generator = QAPairGenerator(quantize=True,mode="pr")
    
    responses = []#qa_generator.generate_batch(df,batch_size=2)
    
    for idx, row in df.iterrows():
        response = qa_generator.generate(row)
        responses.append(response)
    # Flatten nested context into top-level keys
    flat_responses = []
    for r in responses:
        flat = {
            "question": r.get("question"),
            "answer": r.get("answer"),
            "issue_url": r.get("issue_url"),
        }
    
        context = r.get("context", {})
        flat.update({
            "edit_functions": context.get("edit_functions"),
            "problem_statement": context.get("problem_statement"),
            "comments": context.get("comments"),
            "pr_problem_statement": context.get("pr_problem_statement"),
            "pr_comments": context.get("pr_comments"),
        })
    
        flat_responses.append(flat)
    
    # Create the DataFrame
    responses_df = pd.DataFrame(flat_responses)
    print(responses_df)
    for _, response in responses_df.iterrows():
        print(f"Generated question: {response['question']}")
        print(f"Generated answer: {response['answer']}")
        #print(response["edit_functions"])
    responses_df.to_csv("./graph/sklearn/eval_df_w_q.csv",index=False)
  
if __name__ == "__main__":
    main()