from typing import Any, Sequence

import openai
import tiktoken
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

class ChatReadRetrieveReadApproach(Approach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    MAX_HISTORY = 3

    system_message_chat_conversation = """<|im_start|>system Assistant provides accurate information to potential customers of TMBThanachart (TTB) Bank regarding various bank products. These products include accounts, debit cards (both digital and physical), credit cards, insurance, and more. Customers rely on your responses, and any fabrication of data can harm the bank's reputation. Therefore, it is crucial to answer based only on the facts provided in the sources below.

To assist customers effectively, please consider the following:
1.Answer in Thai language: Communicate with customers in Thai language to ensure clear understanding.
2.Use the sources provided below: Refer to the sources provided below to obtain accurate information about TMBThanachart Bank products.
3.Sometimes sources might be empty, so you must say you don't know, dont try to fabricate any information.
4.Sometimes sources might not related to the question, you must say you don't know, don't try to  fabricate any information.
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
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook. 
    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """ Let's think step by step. Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about bank product questions and product benefits and information from factsheets. 
    Generate a search query based on the conversation and the new question. 
    The name "ttb" should always be in lowercase letters.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Search query should be in Thai.


Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
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
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        messages = self.get_messages_from_history(prompt_override=prompt_override, follow_up_questions_prompt=follow_up_questions_prompt,history=history, sources=content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        # new code: new api called object
        # chat_completion = openai.ChatCompletion.create(
        #     deployment_id=self.chatgpt_deployment,
        #     model=self.chatgpt_model,
        #     messages=messages, 
        #     temperature=overrides.get("temperature") or 0.7, 
        #     max_tokens=1500, 
        #     n=1, 
        #     stop=["<|im_end|>", "<|im_start|>"])

        # msg_to_display = '\n\n'.join([str(message) for message in messages])
        # chat_content = chat_completion.choices[0].message.content
        # return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{q}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}
    
        # old implementation
        chat_completion = openai.Completion.create(
            engine=self.chatgpt_deployment,
            prompt=messages,
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1500, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])
        
        return {"data_points": results, "answer": chat_completion.choices[0].text, "thoughts": f"Searched for:<br>{q}<br><br>Conversations:<br>" + messages.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history: Sequence[dict[str, str]], include_last_turn: bool=True, approx_max_tokens: int=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" + "\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot", "") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
    
    def get_messages_from_history(self, prompt_override, follow_up_questions_prompt, history: Sequence[dict[str, str]], sources: str, approx_max_tokens: int = 1000):
        '''
        Generate messages needed for chat Completion api
        '''
        messages = []
        token_count = 0
        # new code: change in promt template
        # if prompt_override is None:
        #     system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        # elif prompt_override.startswith(">>>"):
        #     system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        # else:
        #     system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)
        
        # old code with apdatation
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", sources=sources, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", sources=sources, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(sources=sources, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # messages.append({"role":self.SYSTEM, "content": system_message})
        # token_count += self.num_tokens_from_messages(system_message[-1], self.chatgpt_model)
        
        # latest conversation
        # user_content = history[-1]["user"] + " \nSources:" + sources
        # messages.append({"role": self.USER, "content": user_content})
        # token_count += token_count + self.num_tokens_from_messages(system_message[-1], self.chatgpt_model)

        '''
        Enqueue in reverse order
        if limit exceeds truncate old messages 
        leaving system message behind
        Keep track of token count for each conversation
        If token count exceeds limit, break
        '''
        # for h in reversed(history[:-1]):
        #     if h.get("bot"):
        #         messages.insert(1, {"role": self.ASSISTANT, "content" : h.get("bot")})
        #         token_count += self.num_tokens_from_messages(messages[1], self.chatgpt_model)
        #     messages.insert(1, {"role": self.USER, "content" : h.get("user")})
        #     token_count += self.num_tokens_from_messages(messages[1], self.chatgpt_model)
        #     if token_count > approx_max_tokens*4:
        #         break
        # return messages
        return system_message
    
    def num_tokens_from_messages(self, message: dict[str,str], model: str) -> int:
        """
        Calculate the number of tokens required to encode a message.
        Args:
            message (dict): The message to encode, represented as a dictionary.
            model (str): The name of the model to use for encoding.
        Returns:
            int: The total number of tokens required to encode the message.
        Example:
            message = {'role': 'user', 'content': 'Hello, how are you?'}
            model = 'gpt-3.5-turbo'
            num_tokens_from_messages(message, model)
            output: 11
        """
        encoding = tiktoken.encoding_for_model(self.get_oai_chatmodel_tiktok(model))
        num_tokens = 0
        num_tokens += 2  # For "role" and "content" keys
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
        return num_tokens

    def get_oai_chatmodel_tiktok(self, aoaimodel: str):
        if aoaimodel == "" or aoaimodel is None:
            raise Exception("Expected AOAI chatGPT model name")
        
        return "gpt-3.5-turbo" if aoaimodel == "gpt-35-turbo" else aoaimodel