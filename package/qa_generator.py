import re
import pandas as pd
import torch
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class QAPairGenerator:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", mode:str="issue",quantize=False):
        assert mode in ["issue", "pr"], "Mode must be either 'issue' or 'pr'."
        self.mode = mode
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if quantize:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",quantization_config=bnb_config,torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            return_full_text=False,
        ))

        if mode == "issue":
            self.prompt = PromptTemplate.from_template(self._issue_prompt_template())
        else:
            self.prompt = PromptTemplate.from_template(self._pr_only_prompt_template())

        self.chain = self.prompt | self.llm

    def _issue_prompt_template(self):
        return """You are creating Q&A pairs to evaluate a codebase question-answering system.

Given the following context from an issue and its linked pull requests, generate a **realistic technical question** that could be answered using this information, and a precise answer.

### Context:
- Issue Title and Body (used to generate the question): {problem_statement}
- Functions changed by the PR(s): {edit_functions}
- PR Title and Body (contains attempted fixes): {pr_problem_statement}

### Guidelines:
- Formulate a technical question based on the **issue title and body**.
- Make the question no longer than 1 sentence.
- Use the **functions changed** as a clue to what the answer should focus on.
- Write a precise answer using only the comments and PR content.
- Make the answer at max 1-2 sentences.
- Always end the question with a question mark: "?".
- Always end the answer with a period: ".".
- If no valid question can be made, say:
  Question: Not applicable?
  Answer: Not enough information.

### Output format:
Question: <Your question here?>
Answer: <Your answer here.>
"""  # Your existing issue+pr prompt

    def _pr_only_prompt_template(self):
        return """You are creating Q&A pairs to evaluate a codebase question-answering system.

Use the following context from a GitHub pull request — along with any linked issue details, if available — to generate a realistic, technical question and a specific answer.

### Context
- Functions changed by the PR: {edit_functions}
- PR Title and Description (primary source for question): {pr_problem_statement}
- Issue Title and Body (optional, use to clarify the question if available): {problem_statement}

### Instructions
- Generate a **technical question** based on the **PR title, description, and changed functions**.
- If an issue is provided, use its title/body to **inform or improve the question**, but prioritize PR content.
- Write a **precise answer** using details from **PR comments, reviews, and optionally issue comments**.
- End the question with a **"?"** and the answer with a **"."**.
- If there is not enough information to make a valid Q&A pair, say:
  Question: Not applicable?
  Answer: Not enough information.

### Output format
Question: <Your technical question?>
Answer: <Your accurate and concise answer.>
"""

    def generate(self, issue_data: pd.Series) -> Dict[str, str]:
        try:
            edit_functions = issue_data.get("edit_functions", [])
            print(f"[QAGen] Generating Q&A for issue: {issue_data.get('url', 'Unknown URL')}")
            if len(edit_functions) > 10:
                print("[QAGen] More than 10 edit functions, truncating to 10.")
                edit_functions = edit_functions[:10]
            input_vars = {
                "problem_statement": issue_data.get("problem_statement", ""),
                "edit_functions": ", ".join(edit_functions),
            #    "comments": issue_data.get("comments", ""),
                "pr_problem_statement": issue_data.get("pr_problem_statement", ""),
            #    "pr_comments": issue_data.get("pr_comments", ""),
            }

            result = self.chain.invoke(input_vars)
            question, answer = self._parse_output(result)

            return {
                "question": question,
                "answer": answer,
                "context": input_vars,
                "issue_url": issue_data.get("url", "")
            }

        except Exception as e:
            print(f"[QAGen Error] {e}")
            return {
                "question": None,
                "answer": None,
                "context": {},
                "issue_url": issue_data.get("url", "")
            }

    def generate_batch(self, issues: pd.DataFrame, batch_size: int = 4) -> List[Dict[str, str]]:
        results = []

        for i in range(0, len(issues), batch_size):
            batch_df = issues.iloc[i:i+batch_size]
            batch_inputs = []

            for _, row in batch_df.iterrows():
                batch_inputs.append({
                    "problem_statement": row.get("problem_statement", ""),
                    "edit_functions": ", ".join(row.get("edit_functions", [])),
                    #"comments": row.get("comments", ""),
                    "pr_problem_statement": row.get("pr_problem_statement", ""),
                    #"pr_comments": row.get("pr_comments", ""),
                })

            try:
                #print(batch_inputs)
                torch.cuda.empty_cache()
                outputs = self.chain.batch(batch_inputs)
                print(outputs)
                for input_dict, output_text, (_, row) in zip(batch_inputs, outputs, batch_df.iterrows()):
                    q, a = self._parse_output(output_text)
                    results.append({
                        "question": q,
                        "answer": a,
                        "context": input_dict,
                        "issue_url": row.get("url", "")
                    })

            except Exception as e:
                print(f"[Batch QAGen Error] {e}")
                # Add empty results for failed batch
                for _, row in batch_df.iterrows():
                    results.append({
                        "question": None,
                        "answer": None,
                        "context": {},
                        "issue_url": row.get("url", "")
                    })

        return results

    def _parse_output(self, output: str) -> (str, str):
        # Normalize lines
        lines = output.strip().splitlines()
        question, answer = None, None
    
        # Look for lines that start with question/answer markers
        for line in lines:
            line = line.strip()
            # Match formats like: "**Question:**", "Question:", "- Question:"
            q_match = re.match(r"[*\-]*\s*Question\s*[:\-]*\s*(.*)", line, re.IGNORECASE)
            a_match = re.match(r"[*\-]*\s*Answer\s*[:\-]*\s*(.*)", line, re.IGNORECASE)
    
            if q_match:
                question = q_match.group(1).strip()
            elif a_match:
                answer = a_match.group(1).strip()
    
        # Fallbacks
        if not question:
            question = "Not applicable?"
        elif not question.endswith("?"):
            question = question.rstrip(".") + "?"
    
        if not answer:
            answer = "Not enough information."
        elif not answer.endswith("."):
            answer = answer.rstrip("?") + "."
    
        return question, answer

