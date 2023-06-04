from langchain import PromptTemplate, HuggingFaceHub

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

template = "Question: {question} Answer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceHub(repo_id="google/flan-t5-xl",
                     model_kwargs={"temperature": 1e-10})

question = "When was Google founded?"

response = llm.generate([prompt.template.format(question=question)])

print(response)
