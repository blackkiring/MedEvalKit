def get_multiple_choice_prompt(question,choices,is_reasoning = False,lang = "en"):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    if lang == "en":
        prompt = f"""
Question: {question}
Options: 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + 'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
        else:
            prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    elif lang == "zh":
        prompt = f"""
问题： {question}
选项： 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + '请直接使用给定选项中的选项字母来回答该问题,并将答案包裹在"\\boxed{}"里'
        else:
            prompt = prompt + "\n" +  "请直接使用给定选项中的选项字母来回答该问题。"
    return prompt

def get_judgement_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please output "yes" or "no" and put the answer in one "\\boxed{}".'
        else:
            prompt = question + "\n" + "Please output 'yes' or 'no'(no extra output)."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请输出'是'或'否'，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。"
    return prompt

def get_close_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Answer the question using a single word or phrase and put the answer in one "\\boxed{}".'
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请用一个单词或者短语回答该问题，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请用一个单词或者短语回答该问题。"
    return prompt

def get_open_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please answer the question concisely and put the answer in one "\\boxed{}".'
        else:
            prompt = question + "\n" + "Please answer the question concisely."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请简要回答该问题，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请简要回答该问题。"
    return prompt

def get_report_generation_prompt(image_index_info=""):
    """Generate prompt for report generation tasks.
    
    Args:
        image_index_info: Optional image index information to include in the prompt
        
    Returns:
        Prompt string for report generation
    """
    base_prompt = "You are a helpful assistant."
    if image_index_info:
        base_prompt += f" {image_index_info}"
    base_prompt += " Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
    return base_prompt

def get_image_index_info(num_images):
    """
    Generate image index information for prompts based on the number of images.
    
    Args:
        num_images: Number of images in the sample
        
    Returns:
        String with image index information
    """
    if num_images == 0:
        return ""
    elif num_images == 1:
        return "The index of the given image is 1.\n"
    else:
        indices = ", ".join(str(i) for i in range(1, num_images + 1))
        return f"The indices of the given images are {indices}.\n"

def add_image_index_to_prompt(prompt, image_index_info):
    """
    Add image index information to a prompt before the answer instruction.
    
    This is a helper function to consistently insert image index info into prompts
    across different datasets. It attempts to insert the info before the last line
    (which typically contains the answer instruction).
    
    Args:
        prompt: The original prompt text
        image_index_info: Image index information string to insert
        
    Returns:
        Modified prompt with image index info inserted
        
    Note:
        This function assumes the last line (after the last newline) contains
        the answer instruction. If the prompt doesn't follow this format,
        the image index info will be appended at the end.
    """
    if not image_index_info:
        return prompt
    
    # Try to split before the last newline (where answer instruction usually is)
    parts = prompt.rsplit('\n', 1)
    if len(parts) == 2:
        return parts[0] + '\n' + image_index_info + parts[1]
    else:
        # If no newline found, append at the end
        return prompt + '\n' + image_index_info


