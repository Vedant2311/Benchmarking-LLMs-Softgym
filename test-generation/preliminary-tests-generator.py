import random
import json
import os

def generate_random_pairings_forward_dynamics(corner_bias, horizon, questions_per_config, action_length = "long", options = 50, num_incorrect_options = 3):
    '''
    This function will return: [One input question with action specified] along with [One correct and three incorrect options for next state].
    The four choices that'd be provided are corresponding to randomly picking one image out of all the different random generations available.
    '''
    corner_bias_string = "corner bias" if corner_bias else "random"
    action_length_string = "long horizon" if action_length == "long" else "short horizon"
    dynamics_string = "forward dynamics"
    horizon_string = str(horizon)

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    parent_directory = os.path.dirname(script_directory)
    parent_path = os.path.join(parent_directory, "data", "random", corner_bias_string, horizon_string, action_length_string, dynamics_string)
    
    entries = os.listdir(parent_path)
    num_configs = len([entry for entry in entries if os.path.isdir(os.path.join(parent_path, entry))])
        
    questions = {}
    for config_id in range(num_configs):
        config_parent_path = os.path.join(parent_path, str(config_id))
        questions[config_id]= {}
        for question_id in range(questions_per_config):
            questions[config_id][question_id] = {}

            # Choosing the input question and putting the rgbviz image as the question
            input_question_index = random.randint(0, options - 1)
            questions[config_id][question_id]["question"] = os.path.join(config_parent_path, str(input_question_index), "rgbviz", "0.png")
            questions[config_id][question_id]["options"] = {}

            # Choosing the indices corresponding to the incorrect options
            possible_numbers = list(range(options))
            possible_numbers.remove(input_question_index)
            chosen_numbers = random.sample(possible_numbers, num_incorrect_options)
            chosen_numbers.append(input_question_index)
            random.shuffle(chosen_numbers)

            # Populating options corresponding to the RGB image for them
            correct_option = ""
            for i in range(len(chosen_numbers)):
                letter = chr(65 + i)
                questions[config_id][question_id]["options"][str(letter)] = os.path.join(config_parent_path, str(chosen_numbers[i]), "rgb", "1.png")
                if chosen_numbers[i] == input_question_index:
                    correct_option = str(letter)
            questions[config_id][question_id]["answer"] = correct_option

    with open("random-tests-forward.json", "w") as file:
        json.dump(questions, file, indent = 4)

    print("The JSON file for the preliminary questions in Forward Dynamics generated successfully")

def generate_random_pairings_inverse_dynamics(corner_bias, horizon, questions_per_config = 50, action_length = "long", options = 50, num_incorrect_options = 3):
    '''
    This function creates a Json file corresponding to the questions and answers generated for the Inverse Dynamics questions
    '''
    corner_bias_string = "corner bias" if corner_bias else "random"
    action_length_string = "long horizon" if action_length == "long" else "short horizon"
    dynamics_string = "inverse dynamics"
    horizon_string = str(horizon)

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    parent_directory = os.path.dirname(script_directory)
    parent_path = os.path.join(parent_directory, "data", "random", corner_bias_string, horizon_string, action_length_string, dynamics_string)

    entries = os.listdir(parent_path)
    num_configs = len([entry for entry in entries if os.path.isdir(os.path.join(parent_path, entry))])
        
    questions = {}
    for config_id in range(num_configs):
        config_parent_path = os.path.join(parent_path, str(config_id))
        questions[config_id]= {}
        for question_id in range(questions_per_config):
            questions[config_id][question_id] = {}

            # Choosing the input question and putting the rgb images as the question
            input_question_index = question_id
            questions[config_id][question_id]["initial"] = os.path.join(config_parent_path, str(input_question_index), "rgb", "0.png")
            questions[config_id][question_id]["final"] = os.path.join(config_parent_path, str(input_question_index), "rgb", "1.png")
            questions[config_id][question_id]["options"] = {}

            # Getting the list of option paths to be shuffled
            options_list = []
            for i in range(num_incorrect_options + 1):
                options_list.append(os.path.join(config_parent_path, str(input_question_index), "rgbviz", str(i) + ".png")) 
            random.shuffle(options_list)

            # Populating options corresponding to the RGB image for them
            correct_option = ""
            for i in range(len(options_list)):
                letter = chr(65 + i)
                questions[config_id][question_id]["options"][str(letter)] = options_list[i]
                if options_list[i] == os.path.join(config_parent_path, str(input_question_index), "rgbviz", "0.png"):
                    correct_option = str(letter)
            questions[config_id][question_id]["answer"] = correct_option        

    with open("random-tests-inverse-" + str(action_length) + ".json", "w") as file:
        json.dump(questions, file, indent = 4)

    print("The JSON file for the preliminary questions in Inverse Dynamics generated successfully")

# generate_random_pairings_forward_dynamics(True, 1, 25)
generate_random_pairings_inverse_dynamics(True, 1, 50, "long")