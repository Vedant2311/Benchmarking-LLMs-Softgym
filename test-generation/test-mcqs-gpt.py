def analyze_images_gpt_forward(image_analysis_instruction, question, options, ordering = False):
    '''
    This function takes the paths of the questions and options and returns a predicted choice for each question using GPT-4V
    '''
    import base64
    import requests
    import re
    api_key="sk-proj-6I3KID0QgrjqtOclbfUeT3BlbkFJmpL9PAh5UxTBWJLaAiZW"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Function to append image blocks for payload json
    def append_image_blocks(current_json_block, image_list):
        for image in image_list:
            image_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            }
            current_json_block.append(image_block)
        return current_json_block

    question_image = encode_image(question)
    options_images = [encode_image(option) for option in options]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_analysis_instruction}
                ] + append_image_blocks([], [question_image] + options_images)
            }
        ],
        "max_tokens": 1000,
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    if 'choices' in response:
        text = response['choices'][0]['message']['content']
        print(text)

        if not ordering:
            # Parsing the answer and returning it
            # match = re.search(r'\*\*Answer\*\*:\s*([A-D])', text)
            match = re.search(r'\[([0-9]+)\]', text)
            if match:
                number = int(match.group(1))
                return str(chr(64 + number))
                # return match.group(1)
            else:
                return 'Z'
        else:
            # Use regular expression to find the list part
            match = re.search(r"\[([0-9, ]+)\]", text)
            if match:
                # Extract the matched group (which contains the options)
                options_str = match.group(1)
                # Split the string by comma and strip any whitespace
                options_list = [option.strip() for option in options_str.split(',')]
                return str(chr(64 + int(options_list[0])))
            else:
                return 'Z'  
    else:
        print("Unexpected error occurred")
        return 'Z'

def analyze_images_gpt_inverse(image_analysis_instruction, initial, final, options, ordering = False):
    '''
    This function takes the paths of the questions and options and returns a predicted choice for each question using GPT-4V
    '''
    import base64
    import requests
    import re
    api_key="sk-proj-6I3KID0QgrjqtOclbfUeT3BlbkFJmpL9PAh5UxTBWJLaAiZW"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Function to append image blocks for payload json
    def append_image_blocks(current_json_block, image_list):
        for image in image_list:
            image_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            }
            current_json_block.append(image_block)
        return current_json_block

    initial_image = encode_image(initial)
    final_image = encode_image(final)
    options_images = [encode_image(option) for option in options]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_analysis_instruction}
                ] + append_image_blocks([], [initial_image, final_image] + options_images)
            }
        ],
        "max_tokens": 1000,
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    if 'choices' in response:
        text = response['choices'][0]['message']['content']
        print(text)

        if not ordering:
            # Parsing the answer and returning it
            # match = re.search(r'\*\*Answer\*\*:\s*([A-D])', text)
            match = re.search(r'\[([0-9]+)\]', text)
            if match:
                number = int(match.group(1))
                return str(chr(64 + number))
                # return match.group(1)
            else:
                return 'Z'
        else:
            # Use regular expression to find the list part
            match = re.search(r"\[([0-9, ]+)\]", text)
            if match:
                # Extract the matched group (which contains the options)
                options_str = match.group(1)
                # Split the string by comma and strip any whitespace
                options_list = [option.strip() for option in options_str.split(',')]
                return str(chr(64 + int(options_list[0])))
            else:
                return 'Z'
    else:
        print("Unexpected error occurred")
        return 'Z'

def analyze_images_gemini_forward(image_analysis_instruction, question, options):
    '''
    This function takes the paths of the questions and options and returns a predicted choice for each question using Google's Gemini
    '''
    import base64
    import google.generativeai as genai
    import PIL.Image
    import re

    # Function to encode image
    def encode_image(image_path):
        return PIL.Image.open(image_path)

    api_key = "AIzaSyBJLndqPVSUOta_9gX3E5emyfcKxgUhCKg"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='gemini-pro-vision')
    # question_image = ["question"] + [PIL.Image.open(question)]
    # options_images = []
    # for index, option in enumerate(options):
    #     options_images += encode_image(option, index)
    # encoded_images = question_image + options_images

    generation_config = genai.GenerationConfig(temperature=0.)
    # response = model.generate_content(encoded_images + [image_analysis_instruction], generation_config=generation_config)
    response = model.generate_content([image_analysis_instruction, 'question: ', encode_image(question), 'Option A: ', encode_image(options[0]), 'Option B: ', encode_image(options[1]), 'Option C: ', encode_image(options[2]), 'Option D: ', encode_image(options[3]), 'Please provide your choice and the reason behind the choice.'], generation_config=generation_config)
    text = response.text
    print(text)

    # Parsing the answer and returning it
    match = re.search(r'Answer:\s*([A-D])', text)
    # match = re.search(r"- Answer: (\d+)", text)
    if match:
        # number = int(match.group(1))
        # return str(chr(64 + number))
        return match.group(1)
    else:
        return 'Z'

def analyze_images_gemini_inverse(image_analysis_instruction, initial, final, options):
    '''
    This function takes the paths of the questions and options and returns a predicted choice for each question using Google's Gemini
    '''
    import base64
    import google.generativeai as genai
    import PIL.Image
    import re

    # Function to encode image
    def encode_image(image_path):
        return PIL.Image.open(image_path)

    api_key = "AIzaSyBJLndqPVSUOta_9gX3E5emyfcKxgUhCKg"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='gemini-pro-vision')

    generation_config = genai.GenerationConfig(temperature=0.)
    # response = model.generate_content(encoded_images + [image_analysis_instruction], generation_config=generation_config)
    response = model.generate_content([image_analysis_instruction, 'initial state: ', encode_image(initial), 'final state: ', encode_image(final), 'Option A: ', encode_image(options[0]), 'Option B: ', encode_image(options[1]), 'Option C: ', encode_image(options[2]), 'Option D: ', encode_image(options[3]), 'Please provide your choice and the reason behind the choice.'], generation_config=generation_config)
    text = response.text
    print(text)

    # Parsing the answer and returning it
    match = re.search(r'Answer:\s*Option\s*([A-D])', text)
    # match = re.search(r"- Answer: (\d+)", text)
    if match:
        # number = int(match.group(1))
        # return str(chr(64 + number))
        return match.group(1)
    else:
        match = re.search(r'Answer:\s*([A-D])', text)
        return match.group(1) if match else 'Z'

def analyze_images_random(image_analysis_instruction, question, options):
    '''
    This function returns an answer randomly
    '''
    import random
    def generate_letter_list(length):
        letters = []
        for i in range(length):
            # Generate the letter corresponding to the index
            letter = chr(65 + i)  # 65 is the ASCII value of 'A'
            letters.append(letter)
        return letters
    
    letters = generate_letter_list(len(options))
    return random.choice(letters)

def create_combined_image_forward(question_image_path, option_image_paths, correct_option, predicted_option, output_image_path):
    from PIL import Image, ImageDraw, ImageFont

    # Load the question image
    question_image = Image.open(question_image_path)
    question_width, question_height = question_image.size

    # Load the option images
    option_images = [Image.open(path) for path in option_image_paths]
    option_width, option_height = option_images[0].size

    # Create a new image with enough space for the question and option images
    total_width = max(question_width, option_width * 4)
    total_height = question_height + option_height + 50  # Additional space for annotations

    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Paste the question image at the top center
    question_x = (total_width - question_width) // 2
    new_image.paste(question_image, (question_x, 0))

    # Paste the option images in the second row
    for i, option_image in enumerate(option_images):
        option_x = i * option_width
        new_image.paste(option_image, (option_x, question_height))

    # Create a draw object
    draw = ImageDraw.Draw(new_image)

    # Highlight the correct option
    correct_index = ord(correct_option) - ord('A')
    correct_highlight_x = correct_index * option_width
    correct_highlight_y = question_height
    draw.rectangle([correct_highlight_x, correct_highlight_y, correct_highlight_x + option_width, correct_highlight_y + option_height], outline="red", width=5)

    # Highlight the predicted option
    if predicted_option <= 'D':
        predicted_index = ord(predicted_option) - ord('A')
        predicted_highlight_x = predicted_index * option_width
        predicted_highlight_y = question_height
        draw.rectangle([predicted_highlight_x, predicted_highlight_y, predicted_highlight_x + option_width, predicted_highlight_y + option_height], outline="blue", width=5)

    # Annotate the colors
    font = ImageFont.load_default()
    draw.text((10, total_height - 40), "Red: Actual Answer", fill="red", font=font)
    draw.text((10, total_height - 20), "Blue: Predicted Answer", fill="blue", font=font)

    # Save the new image
    new_image.save(output_image_path)

def create_combined_image_inverse(initial_image_path, final_image_path, option_image_paths, correct_option, predicted_option, output_image_path):
    from PIL import Image, ImageDraw, ImageFont

    # Load the initial and final images
    initial_image = Image.open(initial_image_path)
    final_image = Image.open(final_image_path)
    initial_width, initial_height = initial_image.size
    final_width, final_height = final_image.size

    # Load the option images
    option_images = [Image.open(path) for path in option_image_paths]
    option_width, option_height = option_images[0].size

    # Create a new image with enough space for the initial and final images at the top and option images at the bottom
    total_width = max(initial_width + final_width, option_width * 4)
    total_height = max(initial_height, final_height) + option_height + 50  # Additional space for annotations

    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Paste the initial image at the top left
    new_image.paste(initial_image, (0, 0))

    # Paste the final image at the top right
    new_image.paste(final_image, (initial_width, 0))

    # Paste the option images in the second row
    for i, option_image in enumerate(option_images):
        option_x = i * option_width
        new_image.paste(option_image, (option_x, max(initial_height, final_height)))

    # Create a draw object
    draw = ImageDraw.Draw(new_image)

    # Highlight the correct option
    correct_index = ord(correct_option) - ord('A')
    correct_highlight_x = correct_index * option_width
    correct_highlight_y = max(initial_height, final_height)
    draw.rectangle([correct_highlight_x, correct_highlight_y, correct_highlight_x + option_width, correct_highlight_y + option_height], outline="red", width=5)

    # Highlight the predicted option
    if predicted_option <= 'D':
        predicted_index = ord(predicted_option) - ord('A')
        predicted_highlight_x = predicted_index * option_width
        predicted_highlight_y = max(initial_height, final_height)
        draw.rectangle([predicted_highlight_x, predicted_highlight_y, predicted_highlight_x + option_width, predicted_highlight_y + option_height], outline="blue", width=5)

    # Annotate the colors
    font = ImageFont.load_default()
    draw.text((10, total_height - 40), "Red: Actual Answer", fill="red", font=font)
    draw.text((10, total_height - 20), "Blue: Predicted Answer", fill="blue", font=font)

    # Save the new image
    new_image.save(output_image_path)

image_analysis_instruction_forward = '''
I shall be providing you with five images, where the first image corresponds to the question and the remaining sub-list four images, INDEXED from 1, correspond to possible choices.
In each image, there is a cloth lying on a table which is represented by a background divided into alternating shades of gray.

In the question image, there is also a black arrow which represents an action performed on the cloth by a robot arm.
Essentially - the arm first moves on top of a point on the cloth, lowers itself to pick the cloth at that point, moves up again and travels in the direction of the black arrow, then lowers itself, finally releasing the cloth.
The black arrow represents both the direction along which the robot arm moves as well as the exact length along the surface traversed by the arm.

Your task is to rank the four given options and return an ordering of them, starting from the best to the worst, based on how close they would be from the expected configuration
Return your output in the below format only:
- Answer: <A comma separated ordering of the four choices (for example: [3, 4, 2, 1])>
- Explanation: A justification for your choice by evaluating all the options thoroughly
'''

image_analysis_instruction_inverse = '''
I shall be providing you with six images. The first image corresponds to top-down view of the INITIAL cloth configuration lying flat on a table. The table is represented by a background divided into alternating shades of gray. 
A robot arm performs an action on the cloth by picking a corner and placing it somewhere in the visible view. This results in the FINAL cloth configuration represented by the second image.

The remaining four images correspond to the possible option images for you to choose from. Interpret them as being indexed from 1 to 4. 
In each of these option images, there is a black arrow which represents an action performed on the cloth by a robot arm.
Essentially - the arm first moves on top of a point on the cloth, lowers itself to pick the cloth at that point, moves up again and travels in the direction of the black arrow, then lowers itself, finally releasing the cloth.
The black arrow represents both the direction along which the robot arm moves as well as the exact length along the surface traversed by the arm.

One of these four options corresponds to the correct action that would result in a configuration that matches the FINAL cloth configuration represented in the second image provided to you, as described in the first paragraph of the prompt.
The remaining three options correspond to actions which would not result in the FINAL cloth configuration.
Now, your task is to return an ordering of these four options, starting from the best option to the worst.
Return your output in the below format only:
- Answer: <A comma separated ordering of the four choices (for example: [3, 4, 2, 1])>
- Explanation: A justification for your choice by evaluating all the options thoroughly
'''

# Changing the test-ID based on what is being done
test_id = "GPT-4o-forward-ordering-1"
test_file_name = "random-tests-forward-long.json"
test_type_inverse = False
ordering = True

import json, sys, os
from datetime import date

# Getting the path to the root directory
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
parent_directory = os.path.dirname(script_directory)

date_today = date.today()
with open(os.path.join(parent_directory, "test-generation", "recent-tests", test_file_name), 'r') as json_file:
    tests = json.load(json_file)

# Writing things to the specified log file
output_file_path = os.path.join(parent_directory, "tests", test_id, "logs", str(date_today))
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

correct_count = 0
total_count = 0
if not test_type_inverse:
    for config in tests:
        # Only running for 400 total questions to match the stats with forward dynamics
        if total_count > 400:
            break
        output_file = os.path.join(output_file_path, str(config) + ".log")
        sys.stdout = open(output_file, 'w', buffering=1)

        for question_id in tests[config]:
            # Only running 10 questions per config for now
            if int(question_id) >= 10:
                break
            total_count += 1

            question_json = tests[config][question_id]
            question = question_json["question"]
            options_dict = question_json["options"]
            options = [options_dict[key] for key in options_dict]
            answer = question_json["answer"]

            predicted_answer = analyze_images_gpt_forward(image_analysis_instruction_forward, question, options, ordering)

            output_image_path = os.path.join(parent_directory, "tests", test_id, "test-images", str(date_today), str(config))
            if not os.path.exists(output_image_path):
                os.makedirs(output_image_path)

            create_combined_image_forward(question, options, answer, predicted_answer, os.path.join(output_image_path, str(question_id) + ".png"))

            if predicted_answer == answer:
                correct_count += 1
                print("The model was able to solve the current question successfully")
            else:
                print("The model was unable to solve the current question successfully")
            print("------------------------------------------------------")
else:
    for config in tests:
        # Only running for 400 total questions to match the stats with forward dynamics
        if total_count > 400:
            break
        output_file = os.path.join(output_file_path, str(config) + ".log")
        sys.stdout = open(output_file, 'w', buffering=1)

        for question_id in tests[config]:
            # Only running 10 questions per config for now
            if int(question_id) >= 10:
                break
            total_count += 1

            question_json = tests[config][question_id]
            initial = question_json["initial"]
            final = question_json["final"]
            options_dict = question_json["options"]
            options = [options_dict[key] for key in options_dict]
            answer = question_json["answer"]

            predicted_answer = analyze_images_gpt_inverse(image_analysis_instruction_inverse, initial, final, options, ordering)

            output_image_path = os.path.join(parent_directory, "tests", test_id, "test-images", str(date_today), str(config))
            if not os.path.exists(output_image_path):
                os.makedirs(output_image_path)

            create_combined_image_inverse(initial, final, options, answer, predicted_answer, os.path.join(output_image_path, str(question_id) + ".png"))

            if predicted_answer == answer:
                correct_count += 1
                print("The model was able to solve the current question successfully")
            else:
                print("The model was unable to solve the current question successfully")
            print("------------------------------------------------------")

print("The success rate for the model on this test was:", (correct_count / total_count) * 100)