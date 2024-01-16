from typing import List
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
import os
from dotenv import load_dotenv


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
    
## Question Sample: "what is your favorite food?", "In which year did you buy your car?", "What is the size of your tshirt?", "From where have you completed your engineering?", "where do you park your car?", "when do you wake up in the morning?"

class AllAgents:

    def __init__(self) -> None:
        # self.questions = ["Do you own more than one vehicle?", "Do you always use your turn signals while taking turns?", "Do you drive under the speed limit of 100km/hour?", "Do you talk on the phone while you drive?", "Do you let other people borrow your car?", "Do you have any young drivers in your household?"]

        # self.llm = AzureChatOpenAI(temperature=0.0, 
        #                     openai_api_key= os.getenv("OPENAI_API_KEY"),
        #                     openai_api_base=os.getenv("OPENAI_API_BASE"),
        #                     openai_api_type= os.getenv("OPENAI_API_TYPE"),
        #                     openai_api_version=os.getenv("OPENAI_API_VERSION"), 
        #                     deployment_name=os.getenv("DEPLOYMENT_NAME"))

        #Use the following method of initialising LLM object with helicone

        self.llm = AzureChatOpenAI(
                    headers={
                    "User-Id": f"{os.getenv('USER_ID')}"
                    },
                    temperature=0.0,
                    deployment_name="GPT35",
                    model="gpt-35-turbo",
                )
        pass   

    def get_questions_list(self, problem_statement):

        print("generting ques list")

        prompt_val = f'''Act as a technical investigating officer, who's task is to investigate by asking questions to find the reason of happening of problem or the cause of problem.

        The output should be in the format given below:
        [question, 
        question, ....] 

        for example: 
            problem: The vehicle took double the time to charge
            Prepare list of questions as an investigating officer to find the cause of problem. 
            AI Resp: [
    "Which vehicle, which charger?",
    "When was the charging initiated?",
    "What was the SOC at this time?",
    "When did the charging end?",
    "What was the SOC?",
    "Was there a power failure during this time?",
    "What is the history of the vehicle?",
    "What is the history of the charger?",
    "Prior to this, when was the last time L05 was charged and with which charger?",
    "Charge details?",
    "What were the previous charge details for the charger EC02?",
    "What about subsequent charge for L05?",
    "What about subsequent charge for EC02?",
    "Is there a log of current demanded by vehicle, and the actual current provided by charger for the problematic charge?",
    "Is there a log of errors recorded by the charger?"
]

    Now, prepare list of only 4 questions to solve another problem:
        Problem: {problem_statement}
        Prepare list of questions as a technical investigating officer to find the cause of problem. 
        AI Resp:'''

        # try: 
        #     llm_output = self.llm.predict(text=prompt_val)
        #     print("printing llm output here")
        #     print(llm_output)
        #     generated_questions = eval(llm_output)
        #     if len(generated_questions)>0: 
        #         return generated_questions
        # except Exception as e:
        #     print(e) 
        #     return ["No questions to ask"]

        
        llm_output = self.llm.predict(text=prompt_val)
        print("printing llm output here")
        print(llm_output)
        generated_questions = eval(llm_output)
        print(f"The generated questions are: {generated_questions}")
        if len(generated_questions)>0: 
            return generated_questions
        else:
            return []
    
    def get_specific_questions(self, problem_statement):

        print("generating specific questions")
        if "overheat" in problem_statement:
            test_manual_as_context = '''A primary objective of the thermal performance testing is to demonstrate the ability to meet some fraction of the Peak Power target at various temperatures. The effects of environment (ambient temperature) on device performance will be measured as required by performing the Static Capacity Test and Hybrid Pulse Power Characterization Test (either the low or high HPPC)16 at various temperatures within the operating temperature target range (-30 to +52oC). Typical temperatures for the thermal performance test consist of 52, 0, -10, -20, and -30°C. At the laboratory cell level, such testing has two targets: to characterize the performance of the technology as a function of temperature and to bound the likely constraints on thermal management of full-size cells or batteries. At the module and system level, the emphasis of thermal performance testing is increasingly on thermal management system design and behavior.
                    
        Unless otherwise specified in a device-specific test plan, initial charging should be performed at 30oC during thermal performance testing. This implies a test sequence as follows: (1) fully charge the device to Vmaxop at 30oC; (2) raise or lower the device ambient temperature to the target value; (3) wait a suitable soak period for thermal equalization, typically 4 to 16 hr depending on size and mass; and (4) execute the desired performance test. If self-discharge is a major concern during the soak period, the device can be clamped at a voltage during this period; however, this requires knowledge of the device OCV versus temperature behavior to ensure that the SOC is not changed inadvertently.
                    
        It may be necessary to adjust the rest intervals in the HPPC Test to ensure that thermal stability as well as voltage equilibrium is reached before each repetition of the pulse power characterization profile.
                    
        Complete thermal performance testing is conducted at BOL and at EOL if practical. At middle of life, a thermal performance test at -20°C could be performed as well.'''

        elif "backup" in problem_statement: 
            test_manual_as_context = '''Static Capacity Test
Capacity in ampere-hours and energy in watt-hours removed at the specified constant current discharge rate are reported based on manufacturer-specified termination conditions. (Note that all of this capacity will not generally be useable within USABC operating conditions, and thus it does not reflect conformance to the USABC Available Energy target. However, it is still considered a useful measure of capacity at the laboratory cell stage.) Energy removed (watt-hours) is reported as a function of the percent of operating capacity removed. These data are used for the later calculation of Available Energy. Ampere-hours and watt-hours returned (and the corresponding overall charge/discharge efficiencies) are also reported for the manufacturer-specified charge algorithm.
4.2.1 Capacity Fade
For devices subjected to life testing, the change in capacity from the beginning-of-life value (measured just prior to the start of life testing) to some later point in time is to be reported periodically as Capacity Fade, expressed as a percentage of the original capacity as shown in Equation (2).
 Capacity  CapacityFade(%)=100×1−Capacityt1 
 t0 
where t0 refers to the time of the initial Reference Performance Test (RPT) immediately prior to the start
(2) of life testing and t1 refers to the time of the later RPT where capacity fade is to be determined.
4.2.2 Energy Fade
For devices subjected to life testing, the change in energy from the beginning-of-life value (measured just prior to the start of life testing) to some later point in time is to be reported periodically as Energy Fade, expressed as a percentage of the original energy as shown in Equation (3).
 Energy  EnergyFade(%)=100×1−Energyt1 
 t0 
where t0 refers to the time of the initial Reference Performance Test (RPT) immediately prior to the start
(3) of life testing and t1 refers to the time of the later RPT where capacity fade is to be determined.'''

        elif "discharge" in problem_statement:
            test_manual_as_context = '''Self-Discharge Test
Self-discharge rate is determined over a fixed period (nominally 30 days) at one or more intermediate test conditions (nominally after removing the energy corresponding to the manufacturer-specified margin or 45 kWh for a System Level application). The difference between the energy (watt-hours) measured prior to the test and during the test is considered to be the energy loss reflecting self-discharge during the stand period. This energy loss is computed as the difference between the pretest C/3 energy and the sum of the energies in the partial C/3 discharges before and after the stand period. This value is then divided by the length of the stand period in days and multiplied by the appropriate Battery Size Factor, as shown in Equation (13).
Self Discharge = WhC /3 before test − (Whpart 1 + Whpart 2 ) ×100% (13) WhC/3 before test
The result of this calculation is reported for comparison with the USABC target of less than 1% loss per month. (Note: The self-discharge test for a module with electronic cell balancing circuit etc. should be reported to show the cell self-discharge and any parasitic drain on the module.)
'''

        elif "survey" in problem_statement:
            insurance_survey_questions = self.get_hardcoded_ques()
            return insurance_survey_questions

        else: 
            test_manual_as_context = "generate generic question"
        
        prompt_val = f'''Act as a technical investigating officer, who's task is to investigate by asking questions to find the reason of happening of problem or the cause of problem.

        problem statment: {problem_statement}

        Now give the list of 4 questions considering the following context:
        Given below is the testing manual related to {problem_statement}. Generate questions from the given context that you would ask from the test engineer during the investigation. Your aim should be to find the reason of problem statment. 

        {test_manual_as_context}
        '''
        print(prompt_val)
        llm_output = self.llm.predict(text=prompt_val)
        print(llm_output)
        questions = llm_output.split('\n')

        return questions
    
    def get_hardcoded_ques(self):
        questions = os.getenv('survey_questions')
        ques = eval(questions)
        return ques
         
    def take_answer_agent(self):

        sys_msg_for_answragent = SystemMessage(content='''You will be taking the survey from user in a conversational tone. You will be thanking the user in case he/she answers your question, and before asking the question, you will give a little brief of what you are trying to do. You will analyze the human answer and then try to find the most closest and accurate answer from the human response. If in case, you could not extract any answer in the human response, you will repeat the question until the user gives the appropriate or closest answer. If the human answer is correct, you will return "I got the answer."

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
        Instruction: Ask human this -> Are you currently insured? If so, what type of insurance do you have?
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to simply ask the question but have to give a short brief set up to the question that i am going to ask",
            "Final response": "With this question i would like to know about your insurance cover, and past experiences
            So can you answer: Are you currently insured? If so, what type of insurance do you have?"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: Well last year i bought car insurance
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that he bought car insurance last year, so it means the human is insured with the insurance and type of insurance is 'car insurance'",
            "Final response" : "Okay. So, you are insured and the type of insurance you have is 'Car Insurance'. Thankyou for your answer. I got the answer"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm.
        Instruction: Ask human this -> Have you ever filed a claim with your insurance company? If yes, please briefly describe the experience.
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to ask the question again. I have to give a short brief set up to the question that i am going to ask",
            "Final response": "with this question, i would like to know about your past experince with your insurance company. So can you please answer: Have you ever filed a claim with your insurance company? If yes, please briefly describe the experience. "
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I think last year, in november i met with a minor car accident and in that scenrio i went to the insurance company for claim
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that he went to insurance company for minor car accident claim. So, the human has answered that he filed the claim, but yet has not described his experience.",
            "Final response" : "Thankyou for your response. I can see that you filed the claim for minor accident, but it will be great if you could briefly describe the experience as well (if the claim process was smooth or if it was hard)"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: The process was mostly smooth 
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that the process was mostly smooth. This means that the user filed the claim and it was mostly smooth with his insurance company for minor car accident",
            "Final response" : "Okay. So, you filed the claim for minor accident and the process of claim was mostly smooth'. Thankyou for your answer. I got the answer, let's move on to the next question now."
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        ########## Example 2

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm. written that it is the first question.
        Instruction: Ask human this -> How important do you consider health insurance to be?
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to simply ask the question but have to give a short brief set up to the question that i am going to ask",
            "Final response": "I would like to inquire about your medical insurance-related preferences and experiences. So can you answer: How important do you consider health insurance to be?"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I think health insurance is important and everyone should have it
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that it is important and everyone should have it. This means that the user finds health insurance important",
            "Final response" : "Yes. In today's time health insurance is really important. Thankyou for your answer. I got the answer, let's move on to the next question now."
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm.
        Instruction: Ask human this -> Do you currently have health insurance coverage?
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to ask the question again. I have to give a short brief set up to the question that i am going to ask",
            "Final response": "Sine we are talking about health insurance, with this question, i would like to ask you about your current insurance coverage. So can you please answer: Do you currently have health insurance coverage?"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I have been associated with insurance provider for last 10 months.
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that he is associated with insurance provider for last 10 months. This means that he bought insurance coverage 10 months back and currently he has health insurance coverage",
            "Final response" : "Thankyou for your answer. I can see that you have health insurance coverage for past 10 months. I got the answer."
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''
        Avoid repeating the question to confirm.
        Instruction: Ask human this -> What is your preferred option to get information on your health insurance? 
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''
        {
            "Thought" : "I have to ask the question again. I have to give a short brief set up to the question that i am going to ask",
            "Final response": "I would like to know about your preferences to receive the information. So can you please answer: What is your preferred option to get information on your health insurance?"
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)

        human_msg_for_agent = HumanMessage(content='''Here is the human response. If the answer is there in human response and if it is correct, you will return "I got the answer."
        Human response: I think multiple insurance companies are there in the market.
        Your response:''')

        agent_for_answering.update_messages(message=human_msg_for_agent)

        AI_msg_for_agent = AIMessage(content='''{
            "Thought" : "I have to analyse the human answer and try to find the answer to the question. So the human said that there are multiple companies in the market. This talks about competition, but is no where related to the question that was asked about the preference to receive the information.",
            "Final response" : "Thankyou for your response but i could not understand how multiple companies in the market is related to preference to receive the preference to receive the information. Can you please answer my question again? here is my question: What is your preferred option to get information on your health insurance? "
        }''')

        agent_for_answering.update_messages(message=AI_msg_for_agent)


        return agent_for_answering