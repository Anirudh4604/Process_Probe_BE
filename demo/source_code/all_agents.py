from typing import List
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
import os


class CAMELAgent:

    def __init__(
        self,
        system_message,
        model: AzureChatOpenAI,
        store
    ) -> None:
        self.model = model
        if store == None:
            self.system_message = system_message
            self.init_messages()
            # print("NEW")
        else:
            self.stored_messages = store
            self.system_message = store[0]
            # print("MESSAGES \n",self.stored_messages,"\n SYSTEM MESSAGE \n",self.system_message)

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]
        # for msg in self.stored_messages:
            # print("INTIALIZED",msg.content,"\n")

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        # for msg in self.stored_messages:
            # print("UPDATED",msg.content,"\n")
        return self.stored_messages
    
    def add_messages(self, messgaes): 
        self.stored_messages = messgaes
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        # print("printing the messages here:")
        # print(messages)
        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message.content

    def store_messages(self):
        return self.stored_messages


class AllAgents:

    def __init__(self) -> None:
        # self.questions = ["what is your name?", "How many glasses of water do you drink in a day?", "who is your favorite actor?", "which is your favorite city", "In which year did you buy your car?"]
        self.questions = ['''Which AWS compute service have you found most user-friendly for deploying applications?
                            Options:
                            1. EC2 (Elastic Compute Cloud)
                            2. Lambda
                            3. Elastic Beanstalk
                            4. ECS (Elastic Container Service)''', 
                            '''Which AWS storage service do you prefer for storing and managing large amounts of data?
                            Options:
                            1. S3 (Simple Storage Service)
                            2. EBS (Elastic Block Store)
                            3. Glacier
                            4. EFS (Elastic File System)''',
                            '''Which AWS database service do you find most suitable for your application's data needs?
                            Options:
                            1. RDS (Relational Database Service)
                            2. DynamoDB
                            3. Aurora
                            4. Redshift''',
                            '''Which AWS networking service do you find most effective for managing your application's network resources?
                            Options:
                            1. VPC (Virtual Private Cloud)
                            2. CloudFront
                            3. Route 53 (DNS Service)
                            4. API Gateway''',
                            '''Which AWS security service have you found most reliable in ensuring the safety of your resources?
                            Options:
                            1. IAM (Identity and Access Management)
                            2. Cognito
                            3. KMS (Key Management Service)
                            4. WAF (Web Application Firewall)''']
        # print(f"The list of questions: {self.questions}")
        self.llm = AzureChatOpenAI(temperature=0.0, 
                            openai_api_key= os.getenv("OPENAI_API_KEY"),
                            openai_api_base=os.getenv("OPENAI_API_BASE"),
                            openai_api_type= os.getenv("OPENAI_API_TYPE"),
                            openai_api_version=os.getenv("OPENAI_API_VERSION"), 
                            deployment_name=os.getenv("DEPLOYMENT_NAME"))
        pass    

    def take_answer_agent(self):

        sys_msg_for_answragent = SystemMessage(content='''Your task is to get the answer from human for the given question. You will try to find the most closest and accurate option from the human answer. If in case, you don't find the answer in the human response, you will repeat the question until the user gives the appropriate or closest answer from given options. If the human answer is correct, you will return "I got the answer."

        Your output should be as per the given format                               
        <Format of Output>
        {
          "Thought" : "You will form thought here", 
          "Final response" : "Your final response will come here"                                     
        }
                                               
        ''')

        agent_for_answering = CAMELAgent(sys_msg_for_answragent, self.llm, None)

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm. 
        Instruction: Ask human this -> Which data structure is best suited for implementing a LIFO (Last In, First Out) behavior?
        Options:
                1. Stack
                2. Queue
                3. Linked List
                4. Tree 
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to simply ask the question dircetly",
            "Final response" : "Which data structure is best suited for implementing a LIFO (Last In, First Out) behavior?
        Choose any of the following Options:
                1. Stack
                2. Queue
                3. Linked List
                4. Tree"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I have read about the Stack and Queue. I feel Tree is one datastructure that is best suited.
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to check human response. The human response has answer to the question. From human response I can find the most closest and accurate option human has choosed is option 4. So, i need to respond with 'I got the answer.'",
            "Final response" : "Thank you for answering. I got my answer."
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)


        ########## Example 2

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm. 
        Instruction: Ask human this -> For sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
            1. Merge Sort
            2. Quick Sort
            3. Bubble Sort
            4. Selection Sort 
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to simply ask the question dircetly",
            "Final response" : "For sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
                1. Merge Sort
                2. Quick Sort
                3. Bubble Sort
                4. Selection Sort"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: explain sorting
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to check human response. The human response doesn't include any answer but it has a query from user. I will respond to the query, but i will also ask the question again.'",
            "Final response" : "Sorting is the process of arranging elements in a specific order, often in an ascending or descending sequence, based on certain criteria or rules. It is a fundamental operation in computer science and is used in various applications and algorithms.
                                     
            but can you please answer the question that i asked, 
                                     
            Question: or sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
                1. Merge Sort
                2. Quick Sort
                3. Bubble Sort
                4. Selection Sort"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)


        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I don't get it all, why are you doing this. I don't like it.
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to check human response. The human response doesn't include any answer but it has a statement from the user. I will respond to the statment, but i will also ask the question again.'",
            "Final response" : "Sorry for causing you the inconvenience. I have been asked to take this survey. If you don't want to take this survey you can quit the survey anytime on your own.  

            but since my task is to get the answer from you, can you please answer the question that i asked, 
                                     
            Question: or sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
                1. Merge Sort
                2. Quick Sort
                3. Bubble Sort
                4. Selection Sort"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: Explain all the options to me
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to check human response. The human response doesn't include any answer but it has a statement from the user. I will respond to the statment, but i will also ask the question again.'",
            "Final response" : " Sure. I can give you a brief explain of every point. 
            Merge Sort:
            Merge Sort is a popular comparison-based sorting algorithm that divides an array into smaller subarrays, sorts those subarrays, and then merges them back together to produce a sorted array. It follows the "divide and conquer" approach, where the array is repeatedly divided until it consists of single elements. Then, it merges these smaller arrays while sorting them simultaneously. Merge Sort is known for its stability and consistent performance, making it a reliable choice for sorting large datasets. However, it requires additional memory for the merging process, which can be a drawback in constrained environments.

            Quick Sort:
            Quick Sort is another efficient sorting algorithm that also uses the divide and conquer strategy. It works by selecting a "pivot" element from the array and partitioning the other elements into two subarrays: those less than the pivot and those greater than the pivot. The subarrays are then recursively sorted. Quick Sort is known for its fast average-case performance, often outperforming other sorting algorithms. However, its worst-case performance can be poor if the pivot selection is not well-balanced, leading to a time complexity of O(n^2).

            Bubble Sort:
            Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The pass-through the list is repeated until no swaps are needed, indicating that the list is sorted. While Bubble Sort is straightforward to understand and implement, it is not very efficient, especially for large datasets. Its average and worst-case time complexity is O(n^2), which makes it impractical for sorting large arrays.

            Selection Sort:
            Selection Sort is another elementary sorting algorithm that divides the input list into two parts: the sorted part and the unsorted part. It repeatedly selects the smallest (or largest, depending on the sorting order) element from the unsorted part and moves it to the end of the sorted part. This process continues until the entire list is sorted. Selection Sort is simple to implement and has a consistent time complexity of O(n^2) regardless of the input data. However, it is not very efficient for large datasets, and there are more efficient sorting algorithms available.

            As per your request i have explained all the options but, can you please answer the question that i asked, 
                                     
            Question: or sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
                1. Merge Sort
                2. Quick Sort
                3. Bubble Sort
                4. Selection Sort"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: end the survey, stop the survey
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to check human response. The human response doesn't include any answer but it has a statement from the user. I will respond to the statment, but i will also ask the question again.'",
            "Final response" : " I understand you are not interested in the survey, but this is my job to keep asking the questions. You can click on the "End" button to end or stop the survey, but untill then, i will ask you the question.    
            Question: or sorting a large dataset with a moderate memory footprint, which algorithm do you prefer?
            Options:
                1. Merge Sort
                2. Quick Sort
                3. Bubble Sort
                4. Selection Sort"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        return agent_for_answering