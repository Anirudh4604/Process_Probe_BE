import time
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import os

from .all_agents import AllAgents
from .database import retrieve_table, add_new_entry, add_new_entry_extra_responses, add_new_entry_ques_answers, retrieve_all_questions, add_new_entry_all_questions, retrieve_all_extra_responses, retrieve_all_ques_answers, retrieve_all_generated_questions, add_new_entry_generated_questions, retrieve_all_problem_statements, add_problem_statement, get_all_gen_questions

from .load_model import ModelHuggingFace
import json
from langsmith import Client


model_instance_hf = ModelHuggingFace.load_model_here()



class QuesAnswer:
        
    def __init__(self) -> None:
        load_dotenv()
        client = Client()
        pinecone.init(api_key=os.getenv("PINECONE_API"),
                      environment=os.getenv("PINECONE_ENV"))
        self.all_answers = {}
        self.username = ""
        self.all_generated_questions = []
        self.initialized_all_agents = AllAgents()
        self.agent_for_taking_answers = self.initialized_all_agents.take_answer_agent()
        self.index_name = 'newquesanswer'
        self.index = pinecone.Index(self.index_name)
        pinecone.list_indexes()
        self.VECTOR_DEPLOYMENT_NAME = os.getenv("VECTOR_DEPLOYMENT_NAME")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # self.embed_model = OpenAIEmbeddings(deployment=self.VECTOR_DEPLOYMENT_NAME, chunk_size=1, openai_api_key=self.OPENAI_API_KEY)
        self.embed_model = model_instance_hf
        self.vectorstore = Pinecone(
                            self.index, self.embed_model.embed_query, 'text')
        self.llm = AzureChatOpenAI(
                    headers={
                    "User-Id": f"{os.getenv('USER_ID')}"
                    },
                    temperature=0.0,
                    deployment_name="GPT35",
                    model="gpt-35-turbo",
                )
        
        pass

    def update_data_in_pinecone(self, meta_data, text_embeddings, answer_id):

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=len(text_embeddings[0]),
                metric='cosine'
            )
            # wait for index to finish initialization
            while not pinecone.describe_index(self.index_name).status['ready']:
                time.sleep(1) 
        self.index.upsert(vectors=zip([answer_id], text_embeddings, meta_data), namespace=self.username)

        return None

    def create_answer_embeddings(self, answer_text, question):

        structured_statment = f"The user said {answer_text} when asked {question}"
        
        answer_embeddings = self.embed_model.embed_documents([structured_statment])

        meta_data = [{
        'text': structured_statment,
        'question': question,
        'answer': answer_text,
                }]

        return answer_embeddings, meta_data

    def upload_single_answer_in_vectorDB(self, answer_id, answer_text, question):
        embeddings_of_answers, complete_meta_data = self.create_answer_embeddings(answer_text,question)
        self.update_data_in_pinecone(meta_data=complete_meta_data, text_embeddings=embeddings_of_answers, answer_id=answer_id)

        return None

    def get_similarity_score(self, question):

        docs = self.vectorstore.similarity_search_with_score(
        question,  # the search query
        k=2 
        )

        return docs
    
    def question_review_using_score(self, question): 

        score = self.vectorstore.similarity_search_with_score(query=question, k=1, namespace=self.username)
        print(score)
        if score[0][1] > 0.70: 
            review_result = "Found in vector store"
        if score[0][1] < 0.70: 
            review_result = "Not found in vector store"

        return review_result
    
    def question_review_new(self, question): 
        docs = self.vectorstore.similarity_search(query=question, k=3, namespace=self.username)

        query = f"""Give me the answer of the question from the given context. If you can't find the answer, you will return "Answer not found". 
                   Question: {question}"""

        chain = load_qa_chain(llm=self.llm, chain_type='stuff')
        output = chain({"input_documents": docs, "question":query}, return_only_outputs=True)

        response = output['output_text']

        if "Answer not found" in response or "cannot" in response or "sorry" in response or "no relevant" in response:
            review_result = "Not found in vector store"

        elif "Answer not found" not in response:
            self.all_answers[f'{question}'] = response
            print(f"Question: {question} and Answer:{response}")
            # self.initialized_collect_data.all_ques_answers[f'{question}'] = response

            review_result = "Found in vector store"
        
        return review_result
    
    def new_problem_statement_check(self, username, new_problem_statement):

        print(f"The new problem statement is: {new_problem_statement}")
        previous_problem_statements = retrieve_all_problem_statements(username_val=username)
        print(previous_problem_statements)

        if new_problem_statement not in previous_problem_statements:
            print("Not found in previous problem statements")
            add_problem_statement(username_val=username, problem_statements=[new_problem_statement])
            
            return {"status": True,
                    "remarks": "It is a new problem statement"}
        
        elif new_problem_statement in previous_problem_statements:
            print("Found in previous problem statements")
            
            return {"status": False,
                "remarks": "It is the same problem mentioned again"} 
            
        else: 
            print("Could not get the status")
            return {"status": True,
                    "remarks": "Error! Could not get the status, consider it as a new problem."} 
    
    def get_first_ques(self, username_val, problem_statement):
        # agent_reply = self.initialized_all_agents.questions[0] 
        new_problem_check = self.new_problem_statement_check(username=username_val, new_problem_statement=problem_statement)
        print(new_problem_check)

        if new_problem_check['status'] is False: 
            print("It is an old problem, LLM Shoud not run to generate new questions")
            #This means it is not a new problem. It is an old problem which already exists in the database.
            generated_questions = retrieve_all_generated_questions(username_val=username_val, problem_statement=problem_statement)
            print(generated_questions)
        
            if len(generated_questions)>1:
                print(f"Retrieved generated question in get_first_ques {generated_questions}") 
                self.all_generated_questions = generated_questions
                return generated_questions[0]
            
            elif len(generated_questions)<1: 
                print("generating ques from get_questions_list")
                # There are three methods of generating questions: 
                # 1. Generic method -- .get_questions_list
                # 2. RAG-based Ashok Leyland Questions -- .get_specific_questions
                # 3. Pre-defined list of ques -- .get_hardcoded_ques
                
                # 1. Generic method -- .get_questions_list
                #generated_questions = self.initialized_all_agents.get_questions_list(problem_statement=os.getenv("PROBLEM_STATEMENT_2"))

                # 2. RAG-based Ashok Leyland Questions -- .get_specific_questions
                generated_questions = self.initialized_all_agents.get_specific_questions(problem_statement=problem_statement)
                
                # 3. Pre-defined list of ques -- .get_hardcoded_ques
                # generated_questions = self.initialized_all_agents.get_hardcoded_ques()
                
                print(f"received {generated_questions}")
                print(type(generated_questions))
                add_new_entry_generated_questions(all_generated_questions=generated_questions, username_val=username_val, problem_statement=problem_statement)
                self.all_generated_questions = generated_questions
                return generated_questions[0]
        
        elif new_problem_check['status'] is True: 
            print("It is a new problem. LLM should be called.")
            # There are three methods of generating questions: 
            # 1. Generic method -- .get_questions_list
            # 2. RAG-based Ashok Leyland Questions -- .get_specific_questions
            # 3. Pre-defined list of ques -- .get_hardcoded_ques
            
            # 1. Generic method -- .get_questions_list
            #generated_questions = self.initialized_all_agents.get_questions_list(problem_statement=os.getenv("PROBLEM_STATEMENT_2"))

            # 2. RAG-based Ashok Leyland Questions -- .get_specific_questions
            generated_questions = self.initialized_all_agents.get_specific_questions(problem_statement=problem_statement)
            
            # 3. Pre-defined list of ques -- .get_hardcoded_ques
            # generated_questions = self.initialized_all_agents.get_hardcoded_ques()
            
            print(f"received {generated_questions}")
            print(type(generated_questions))
            add_new_entry_generated_questions(all_generated_questions=generated_questions, username_val=username_val, problem_statement=problem_statement)
            self.all_generated_questions = generated_questions
            return generated_questions[0]
    
    def get_next_ques(self, previous_question, username_val):

        generated_questions = get_all_gen_questions(username_val=username_val, previous_ques=previous_question)
        self.all_generated_questions = generated_questions

        element_to_find = previous_question
        index = generated_questions.index(element_to_find)

        try: 
            agent_reply = generated_questions[index+1]
        except Exception as e: 
            if "list index out of range" in str(e):
                agent_reply = "All questions asked"
            else: 
                print("Got error in fetching next question")
                print(e)

        if "All questions asked" in agent_reply or "All questions have been asked" in agent_reply or "All questions" in agent_reply:

            return "All questions asked"
        

        elif "All questions asked" not in agent_reply or "All questions have been asked" not in agent_reply or "All questions" not in agent_reply:
            print("Sending question for review...")
            ques_review = self.question_review_new(agent_reply)
            print("Review done")
            # print(ques_review)
            
            if "Found in vector store" in ques_review:
                print("It was found in vector store")
                print("Getting another question")
                return self.get_next_ques(previous_question=agent_reply, username_val=username_val)
                
            elif "Not found in vector store" in ques_review:
                print("Not found in vector store.")
                return agent_reply
            
    def get_current_ques_answer_status(self, username_val):

        total_questions = self.all_generated_questions
        existing_ques_answer = retrieve_all_ques_answers(username_val=username_val)
        answered_questions = list(existing_ques_answer.keys())

        return  {"list of all questions":total_questions,
                 "list of answered questions":answered_questions,
                 "total number of questions":len(total_questions),
                 "total number of answered questions":len(answered_questions)
                 }


class GetSurveyAnswersFromUser:

    def __init__(self, username) -> None:
        self.initialized_utils = QuesAnswer()
        self.initialized_utils.username = username
        self.all_ques_answers = {} #all answers that were necessary for questions will come here mapped with their parent question 
        self.extra_responses = [] #all extra answers will come here
        self.human_answers = [] 
        self.all_questions = []

        pass

    def qna_from_user(self, username_val):

        ques_input = self.initialized_utils.get_first_ques(username_val=username_val)

        human_msg_for_answragent = HumanMessage(content=f"""Instruction: Ask human this -> {ques_input} 
                                                            Your question:""")

        answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)

        print(answr_agent_reply)

        human_answr_input = ""

        for i in range(20):

            if len(human_answr_input)>1:
                if human_answr_input not in self.human_answers:
                    print(f"The user answered {human_answr_input} for question {ques_input}")
                    self.initialized_utils.upload_single_answer_in_vectorDB(answer_id=f"{ques_input}_{human_answr_input}",
                                                                            answer_text=human_answr_input,
                                                                            question=ques_input)
                    self.human_answers.append(human_answr_input)

            if "I got the answer" in answr_agent_reply:

                self.all_ques_answers[f'{ques_input}'] = human_answr_input
                self.initialized_utils.all_answers[f'{ques_input}'] = human_answr_input
            
                # ques_input = input("give question that will come from agent 1:")
                print("getting next question.......")
                new_question = self.initialized_utils.get_next_ques(previous_question=ques_input, username_val=username_val)
                ques_input = new_question

                if "All questions asked" not in ques_input:

                    human_msg_for_answragent = HumanMessage(content=f"""Instruction: Ask human this -> {ques_input} 
                                                                Your question:""")
                    
                    answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)
                    print(answr_agent_reply)
                
                else: 
                    print("All questions asked")
                    break

            else: 

                if human_answr_input not in self.all_ques_answers.values():
                    self.extra_responses.append(human_answr_input)

                human_answr_input = input("give human answer: ")

                if "stop" in human_answr_input:
                    print("Human force stop")
                    break
                
                human_msg_for_answragent = HumanMessage(content=f"""Here is the human answer. If the answer is correct, you will return "I got the answer."
                                                                    Human answer: {human_answr_input}
                                                                    Your response:""")

                answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)
                print(answr_agent_reply)
            
        
        return self.initialized_utils.all_answers
    

    def get_response_for_survey_ques(self, human_text=None, user_name_val=None, prob_statement=None):

        if human_text is None and prob_statement is not None:
            ques_input = self.initialized_utils.get_first_ques(username_val=user_name_val, problem_statement=prob_statement)
            self.all_questions.append(ques_input)
            human_msg_for_answragent = HumanMessage(content=f"""Avoid repeating the question to confirm. written that it is the first question.

                                                            Instruction: Ask human this -> {ques_input}

                                                            Your question:""")

            print(human_msg_for_answragent)
            answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)
            store_messages = self.initialized_utils.agent_for_taking_answers.store_messages()
            add_new_entry(user_name=user_name_val, messages=store_messages)
            add_new_entry_all_questions(all_questions=self.all_questions, username_val=user_name_val)
            current_status = self.initialized_utils.get_current_ques_answer_status(username_val=user_name_val)
            print(answr_agent_reply)
            return {
                "llm_response": answr_agent_reply,
                "current_status": current_status
                }

        if human_text is not None:

            all_retrieved_questions = retrieve_all_questions(username_val=user_name_val)
            current_question = all_retrieved_questions[-1]
            existing_messages = retrieve_table(user_name=user_name_val)
            if len(existing_messages) > 0:
                self.initialized_utils.agent_for_taking_answers.add_messages(messgaes=existing_messages)
                human_msg_for_answragent = HumanMessage(content=f"""Here is the human answer. If the answer is correct, you will return "I got the answer."

                                                                    Human answer: {human_text}

                                                                    Your response:""")
                prev_answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)

                self.initialized_utils.upload_single_answer_in_vectorDB(answer_id=f"{current_question}_{human_text}",

                                                                            answer_text=human_text,

                                                                            question=current_question)
                store_messages = self.initialized_utils.agent_for_taking_answers.store_messages()
                add_new_entry(user_name=user_name_val, messages=store_messages)
                if "I got the answer" in prev_answr_agent_reply:
                        self.all_ques_answers[f'{current_question}'] = human_text
                        self.initialized_utils.all_answers[f'{current_question}'] = human_text

                        # ques_input = input("give question that will come from agent 1:")
                        print("getting next question.......")
                        new_question = self.initialized_utils.get_next_ques(previous_question=current_question, username_val=user_name_val)
                        ques_input = new_question
                        add_new_entry_ques_answers(ques_answers=self.initialized_utils.all_answers,

                                                   username_val=user_name_val)

                        if "All questions asked" not in ques_input:
                            self.all_questions.append(ques_input)
                            add_new_entry_all_questions(all_questions=self.all_questions, username_val=user_name_val)
                            human_msg_for_answragent = HumanMessage(content=f"""Avoid repeating the question to confirm.

                                                                        Instruction: Ask human this -> {ques_input}

                                                                        Your question:""")
                            answr_agent_reply = self.initialized_utils.agent_for_taking_answers.step(human_msg_for_answragent)
                            store_messages = self.initialized_utils.agent_for_taking_answers.store_messages()
                            add_new_entry(user_name=user_name_val, messages=store_messages)
                            print(answr_agent_reply)
                            # retrive_thankyou_statement = prev_answr_agent_reply.split()
                            current_status = self.initialized_utils.get_current_ques_answer_status(username_val=user_name_val)
                            return {
                                    "llm_response": prev_answr_agent_reply + answr_agent_reply,
                                    "current_status": current_status
                                    }

                        else:
                            print("All questions asked")
                            current_status = self.initialized_utils.get_current_ques_answer_status(username_val=user_name_val)
                            return {
                                    "llm_response": "All questions asked",
                                    "current_status": current_status
                                    }

                else:

                    if human_text not in self.all_ques_answers.values():
                        self.extra_responses.append(human_text)
                        add_new_entry_extra_responses(extra_responses=self.extra_responses,

                                                      username_val=user_name_val)
                    current_status = self.initialized_utils.get_current_ques_answer_status(username_val=user_name_val)
                    return {
                            "llm_response": prev_answr_agent_reply,
                            "current_status": current_status
                            }

    def get_structured_ques_answers(self, user_name_val): 
        existing_ques_answers = retrieve_all_ques_answers(username_val=user_name_val) 
        existing_answers= list(existing_ques_answers.values())
        existing_questions = list(existing_ques_answers.keys())
        # existing_questions = retrieve_all_questions(username_val=user_name_val) 
        if len(existing_ques_answers)>0: 
            ques = existing_questions 
            ans = existing_answers 

            output = {} 
            for i in range(min(len(ques),len(ans))): 
                output[f"Query{i+1}"] = {"question":ques[i], "answer":ans[i]} 
            # output = [{"question": question, "answer": answer} for question,answer in zip(existing_questions, existing_answers)]
            return output
        else: 
            return [] 
        

    def get_structured_extra_responses(self, user_name_val): 
        existing_extra_responses = retrieve_all_extra_responses(username_val=user_name_val) 
        if len(existing_extra_responses)>1: 
            prompt_for_extra_ans = f""" You have to create the questions for the given list of answers. There are total {len(existing_extra_responses)} answers, thus you have to create {len(self.extra_responses)} questions, for every given answer. 
                The list of answers are given below, delimited by hashtags #. 
                List of answers: #{existing_extra_responses}# 
                <Formating> 
                The output should be a Markdown code snippet formatted in the following 
                schema. 
                {{ 
                "Create question for the given answer. The question will come here": "answer will come here", 
                "Create question for the given answer. The question will come here": "answer will come here", 
                "Create question for the given answer. The question will come here": "answer will come here", 
                "Create question for the given answer. The question will come here": "answer will come here", 
                and so on. 
                }} 

            """ 

            response_for_extra_ans = self.initialized_utils.llm.predict(prompt_for_extra_ans) 
            extra_data = json.loads(response_for_extra_ans) 
            extra_ques = extra_data.keys() 
            extra_ans = extra_data.values() 
            extra_ques_list = [] 
            extra_ans_list = [] 

            for i in extra_ques: 
                extra_ques_list.append(i) 
            for i in extra_ans: 
                extra_ans_list.append(i) 
                
            output = {} 
            for i in range(len(extra_ques_list)): 
                output[f"Extra_response{i+1}"] = {"question":extra_ques_list[i], "answer":extra_ans_list[i]} 
            return output 
        else: 
            return [] 

