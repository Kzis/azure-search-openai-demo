import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    MAX_HISTORY = 3

    prompt_prefix = """<|im_start|>system Assistant provides accurate information to potential customers of TMBThanachart (TTB) Bank regarding various bank products. These products include accounts, debit cards (both digital and physical), credit cards, insurance, and more. 

To assist customers effectively, please consider the following:
1.Answer in Thai language: Communicate with customers in Thai language to ensure clear understanding.
2.Use the sources provided below: Refer to the sources provided below to obtain accurate information about TMBThanachart Bank products.
3.Sometimes sources might be empty, so you must say you don't know, do not try to fabricate any information.
4.Sometimes sources might not relate to the question, you must say you don't know, don't try to fabricate any information.
5.Seek clarification if needed: If additional context or specific product details would help provide better assistance, feel free to ask clarifying questions.
6.When you site any information you should always end your message with: This AI is still in development and may contain errors or limitations in the accuracy of the information provided. Please use the information with caution and rely on your own judgment.
7.Be brief with your answer
8.You must strictly give the answer based on provide souce below, You can only say what is in the source, you must not fabricate any sources. you don't have access to internet, therefore you must not say or cite any website.
9.You are a man, speak like a man, not a woman, use word like ผม, ครับ. not ฉัน, คะ, ค่ะ. 
Remember, your goal is to provide reliable information based on the provided sources and assist potential customers with their inquiries regarding TMBThanachart Bank's range of products.
##########
Sources:
{sources}
##########
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response.  Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

{follow_up_questions_prompt}
{injected_prompt}

<|im_end|>
{chat_history}
"""

    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about product benefits and information from factsheets. 
    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Let's think step by step. Below is a new question asked by the user that needs to be answered by searching in a knowledge base about bank product questions and product benefits and information from factsheets. 
After read a question, categorize whether the question is in one of these category: ttb all free, ttb touch. Description of these categories are provided below. Then, generate a search query based on the question.
The name "ttb" in the search query should always be in lowercase letters.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Search query should be in Thai.
###
ttb touch: Bank's application which used to transfer, save, loan money. Choose this when asked for instruction on how to use the application to loan automobile, register new account, look into insurance, register new into new bank product, authentication and many more. ttb product eg. ttb flash, so goood, cash2go.
ttb all free: Choose this when question asking about ttb all free benefits and speification. For example: debit and virtual debit card of ttb all free, fees, and other benefits receive from ttb all free account.
###
Question: สมัครบัญชีใหม่ทำอย่างไร
Search query: ขั้นตอนการสมัคร ttb touch, Category: ttb touch

Question: ttb all free ดีอย่างไร
Search query: สิทธิประโยชน์ของ ttb all free, Category: ttb all free
###

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client1: SearchClient, search_client2: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client1 = search_client1
        self.search_client2 = search_client2
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict) -> any:
        history_length = len(history)
        if history_length > self.MAX_HISTORY:
            history = history[-self.MAX_HISTORY:]
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=None, question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=120, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            for_search = q.split(", ")
            q = for_search[0]
            cate = for_search[1].split(": ")
            if cate[1] == "ttb touch":
                r = self.search_client1.search(q, filter=filter, top=top)
                print("ttb touch")
            elif cate[1] == "ttb all free":
                r = self.search_client2.search(q, filter=filter, top=top)
                print("ttb all free")
            else:
                print("Unknow category")
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1500, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=0) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text