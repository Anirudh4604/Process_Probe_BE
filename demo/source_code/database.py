from demo.models import LLMMessages, ConversationHistory, ProblemsTracker
from langchain.schema import (HumanMessage, AIMessage, SystemMessage)


def fetch_all_usernames():
    # Use the Django ORM to query the LLMMessages model for usernames
    usernames = LLMMessages.objects.values_list('username', flat=True)
    # print(list(usernames))
    return list(usernames)


def update_table(user_name, new_messages):
    try:
        message = LLMMessages.objects.get(username=user_name)
        message.messages = new_messages
        message.save()
        return True
    except LLMMessages.DoesNotExist:
        return False
    

def add_new_entry(user_name, messages):
    message_list = str([message.content for message in messages])

    all_users = fetch_all_usernames()

    if user_name not in all_users:
        print("It is a new user")
        # print(user_name)
        # print(all_users)
        new_messages = message_list
        # Create a new LLMMessages object and save it to the database
        new_entry = LLMMessages(username=user_name, messages=new_messages)
        new_entry.save()
    else:
        print("User already registered before")
        update_table(user_name=user_name, new_messages=message_list)


def retrieve_table(user_name):
    try:
        message_entry = LLMMessages.objects.get(username=user_name)
        messages = eval(message_entry.messages)
        
        all_messages = []

        for i, msg_content in enumerate(messages):
            i = i + 1
            if i == 1:
                all_messages.append(SystemMessage(content=msg_content))
            elif i % 2 == 0:
                all_messages.append(HumanMessage(content=msg_content))
            else: 
                all_messages.append(AIMessage(content=msg_content))

        return all_messages

    except LLMMessages.DoesNotExist:
        return []
    

def add_new_entry_extra_responses(extra_responses, username_val):
    try:
        # Check if there is an existing entry with extra_responses for the username
        existing_entry = ConversationHistory.objects.filter(username=username_val, extra_responses__isnull=False).exists()

        if existing_entry:
            # Update existing entry
            existing_record = ConversationHistory.objects.get(username=username_val)
            # print(existing_record.extra_responses)
            if len(existing_record.extra_responses)>1:
                existing_extra_responses = eval(existing_record.extra_responses)
                combined_extra_resp = existing_extra_responses + extra_responses
                existing_record.extra_responses = str(combined_extra_resp)
                existing_record.save()
            else: 
                existing_record.extra_responses = str(extra_responses)
                existing_record.save()

        else:
            # Check if there is any entry for the username
            existing_record = ConversationHistory.objects.filter(username=username_val).first()

            if existing_record:
                # Update existing entry
                existing_record.extra_responses = str(extra_responses)
                existing_record.save()
            else:
                # Create a new entry
                new_entry = ConversationHistory(username=username_val, extra_responses=str(extra_responses))
                new_entry.save()

    except Exception as e:
        print(f"An error occurred: {e}")

    return None


def add_new_entry_ques_answers(ques_answers, username_val):
    try:
        # Check if there is an existing entry with messages for the username
        existing_entry = ConversationHistory.objects.filter(username=username_val, messages__isnull=False).exists()

        if existing_entry:
            print("there is an existing extry for username and messages")
            # Update existing entry
            existing_record = ConversationHistory.objects.get(username=username_val)
            print(f"messages got here: {existing_record.messages}")
            if len(existing_record.messages)>1: 
                existing_ques_answers = eval(existing_record.messages)
                combined_ques_answers = {**existing_ques_answers, **ques_answers}
                existing_record.messages = str(combined_ques_answers)
                existing_record.save()
            
            else: 
                existing_record.messages = str(ques_answers)
                existing_record.save()

        else:
            # Check if there is any entry for the username
            existing_record = ConversationHistory.objects.filter(username=username_val).first()

            if existing_record:
                print("there is an existing extry for just username")
                # Update existing entry
                existing_record.messages = str(ques_answers)
                existing_record.save()
            else:
                print("No existing extry")
                # Create a new entry
                new_entry = ConversationHistory(username=username_val, messages=str(ques_answers))
                new_entry.save()

    except Exception as e:
        print(f"An error occurred: {e}")

    return None



def add_new_entry_all_questions(all_questions, username_val):

    try:
        # Check if there is an existing entry with extra_responses for the username
        existing_entry = ConversationHistory.objects.filter(username=username_val, all_questions__isnull=False).exists()

        if existing_entry:
            # Update existing entry
            existing_record = ConversationHistory.objects.get(username=username_val)
            # print(existing_record.all_questions)
            if len(existing_record.all_questions)>1:
                existing_all_questions = eval(existing_record.all_questions)
                combined_extra_ques = existing_all_questions + all_questions
                existing_record.all_questions = str(combined_extra_ques)
                existing_record.save()
            else: 
                existing_record.all_questions = str(all_questions)
                existing_record.save()

        else:
            # Check if there is any entry for the username
            existing_record = ConversationHistory.objects.filter(username=username_val).first()

            if existing_record:
                # Update existing entry
                existing_record.all_questions = str(all_questions)
                existing_record.save()
            else:
                # Create a new entry
                new_entry = ConversationHistory(username=username_val, all_questions=str(all_questions))
                new_entry.save()

    except Exception as e:
        print(f"An error occurred: {e}")

    return None


def add_new_entry_generated_questions(all_generated_questions, username_val, problem_statement):    
    try:
        new_entry = ProblemsTracker(username=username_val, generated_ques=str(all_generated_questions), problems=problem_statement)
        new_entry.save()
        return all_generated_questions

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def retrieve_all_generated_questions(username_val, problem_statement):

    try: 
        generated_ques = ProblemsTracker.objects.filter(username=username_val, problems=problem_statement)
        generated_ques_list = list(generated_ques.values_list('generated_ques', flat=True))
        print(f"existing entry: {generated_ques_list}")
        # exiting_generated_questions = existing_entry.problems
        return eval(generated_ques_list[0])

    except Exception as e: 
        print(e)
        return []
    
def get_all_gen_questions(username_val, previous_ques):
    generated_ques_entries = ProblemsTracker.objects.filter(username=username_val).values_list('generated_ques', flat=True)
    # Converting the result to a list (if needed)
    generated_ques_list = list(generated_ques_entries)
    # Printing the result
    if generated_ques_list:
        print(f"Generated Questions for {username_val}: {generated_ques_list}")
        previous_question = previous_ques
        for each_ques_list in generated_ques_list: 
            for each_ques_string in eval(each_ques_list): 
                if previous_question in each_ques_string:
                    print("found")
                    return eval(each_ques_list)
                else: 
                    print("Not found")
                    pass
    else:
        print(f"No generated questions found for {username_val}. Check 'get_all_gen_questions' to debug.")
        return []
    

def retrieve_all_questions(username_val):
    
    existing_record = ConversationHistory.objects.get(username=username_val)

    if len(existing_record.all_questions)>1:

        existing_all_questions = eval(existing_record.all_questions)

        
        return existing_all_questions
    
    else: 
        return []
    

def retrieve_all_extra_responses(username_val): 
    existing_record = ConversationHistory.objects.get(username=username_val) 
    if len(existing_record.extra_responses)>1: 
        existing_all_extra_responses = eval(existing_record.extra_responses) 
        return existing_all_extra_responses 
    else: 
        return [] 

def retrieve_all_ques_answers(username_val): 
    existing_record = ConversationHistory.objects.get(username=username_val) 
    if len(existing_record.messages)>1: 
        existing_ques_answers = eval(existing_record.messages) 
        return existing_ques_answers 
    else: 
        return {}


def add_problem_statement(username_val, problem_statements):

    try:
        # Check if there is an existing entry with problem_statment for the username
        existing_entry = ConversationHistory.objects.filter(username=username_val, problem_statements__isnull=False).exists()

        if existing_entry:
            # Update existing entry
            existing_record = ConversationHistory.objects.get(username=username_val)
            existing_all_problem_statements = eval(existing_record.problem_statements)
            combined_all_problem_statements = existing_all_problem_statements + problem_statements
            existing_record.problem_statements = str(combined_all_problem_statements)
            existing_record.save()
            return combined_all_problem_statements

        else:
            # Check if there is any entry for the username
            existing_record = ConversationHistory.objects.filter(username=username_val).first()

            if existing_record:
                # Update existing entry
                existing_record.problem_statements = str(problem_statements)
                existing_record.save()
                return problem_statements
            else:
                # Create a new entry
                new_entry = ConversationHistory(username=username_val, problem_statements=str(problem_statements))
                new_entry.save()
                return problem_statements

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def retrieve_all_problem_statements(username_val):

    try: 
        existing_record = ConversationHistory.objects.get(username=username_val)

        if len(existing_record.problem_statements)>1 or len(existing_record.problem_statements)==1:

            existing_problem_statements = eval(existing_record.problem_statements)

            return existing_problem_statements
        
        else: 
            return []
    except: 
        return []