import datasets, json, re
from collections import defaultdict

input_dir = '/path/to/scratch/data/processed/'

def make_mcq_prompt(
    question,
    choices,
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_format or label_prefix + "A."
    if "A" not in label_format:
        raise ValueError(
            "Label format must contain 'A' to indicate where to place the answer label"
        )
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join(
        [
            f"{label_format.replace('A', label)} {text}"
            for label, text in zip(choice_labels, choices)
        ]
    )
    
    prompt = f"{question_prefix}{question}\n{choices_prefix}{choices_text}\n{answer_prefix}"
    return prompt

def make_mcq_prompt_without_input(
    question,
    choices,
    question_prefix="Question: ",
    choices_prefix="",
    answer_prefix="Answer:",
    label_prefix=" ",
    label_format=None,
):
    choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_format = label_format or label_prefix + "A."
    if "A" not in label_format:
        raise ValueError(
            "Label format must contain 'A' to indicate where to place the answer label"
        )
    # Use prefix space before each answer label to avoid ambiguity in tokenization
    choices_text = "\n".join(
        [
            f"{label_format.replace('A', label)} {text}"
            for label, text in zip(choice_labels, choices)
        ]
    )
    prompt = f"{answer_prefix}"
    return prompt
'''
Formatter for dream QA
'''
def dreamFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    input = example['inputs_pretokenized'].split('-')
    question = input[0]
    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices, question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question, choices, question_prefix="")
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
'''
Formatter for qasc QA. It is 8-way MCQ task with a focus on sentence composition

{'answer_choices': ['pearls', 'streams', 'shells', 'diamonds', 'rain', 'beads', 'cooled', 'liquid'],
'inputs': [493, 9, 26, 7, 13, 387, 33, 5147, 57, 387, 3, 11373, 975, 537, 7, 53, 6, 11, 15328, 33, 263, 13,\
    387, 3, 11373, 5, 9246, 175, 6688, 6, 125, 686, 13, 387, 3239, 19, 5147, 57, 15328, 859, 8, 826, 931, 10,\
        3, 18, 18447, 7, 3, 18, 15841, 3, 18, 7300, 7, 3, 18, 7097, 7, 3, 18, 3412, 3, 18, 18459, 3, 18, 3, 22964,\
            3, 18, 4400],
'inputs_pretokenized': 'Beads of water are formed by water vapor condensing, and clouds are made of water vapor.\
    Given these facts, what type of water formation is formed by clouds among the following options:\n\
        - pearls\n - streams\n - shells\n - diamonds\n - rain\n - beads\n - cooled\n - liquid\n\n',\
            'targets': [18459, 1], 'targets_pretokenized': ' \n\n      beads   '}
'''
def qascFormatter(example, input_flag=True):
    
    fact1 = example['fact1']
    fact2 = example['fact2']
    question = fact1.rstrip('.')+', and '+fact2+' Given these facts, '+example['question'].rstrip().rstrip('?')+' among the following options:'
    choice_labels = example['choices']['label']
    choices = example['choices']['text']
    answer = example['answerKey'].strip()
    
    label_index = choice_labels.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices, question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question, choices, question_prefix="")
    gold = example['answerKey'] +". "+choices[label_index]
    
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
'''
Formatter for commonsense QA
'''
def cosFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    input = example['inputs_pretokenized'].split('-')
    question = input[0]
    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices)
    else:
        query = make_mcq_prompt_without_input(question, choices)
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value

'''
Formatter for qaAIL QA. 
It also has annotation for question type which I have not used for now. 
But can be used later for difficulty
'''
def quailFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    input = example['inputs_pretokenized'].split('\n-')
    question = input[0]
    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices)
    else:
        query = make_mcq_prompt_without_input(question, choices)
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
def remove_q_prefix(text):
    return re.sub(r'^\s*Q[\.:]?\s*', '', text, flags=re.IGNORECASE)

def sciqFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    question = remove_q_prefix(example['inputs_pretokenized'].split('Choices:\n\n-')[0])
    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices)
    else:
        query = make_mcq_prompt_without_input(question , choices)
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
'''
Formatter for cosmos QA
'''
def cosmosFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    question = example['inputs_pretokenized']

    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices, question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question, choices, question_prefix="")
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value
'''
Formatter for wiqa 
It targets what if reasoning
I have sourced it directly from hugginface instead of bigscience repo because the 
formatting is easier
'''
def wiqaFormatter(example, input_flag=True):
    #print('wiqaFormatter ',example)
    paragraph = '. '.join(example['question_para_step'])
    
    if input_flag:
        query = make_mcq_prompt(paragraph+ " " + example['question_stem'], example['choices']['text'])
    else:
        query = make_mcq_prompt_without_input(paragraph+ " " + example['question_stem'], example['choices']['text'])
    gold = example['answer_label_as_choice'] +". "+example['answer_label']
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]

    return value
'''
Formatter for wikihop
It is  a bit tricky. 
Some questions have more than 26 options
'''

def wikiHopFormatter(example, input_flag=True):
    
    num_choices = len(example['answer_choices'])
    question = example['inputs_pretokenized']
    answer = example['targets_pretokenized'].strip()
    '''
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices, question_prefix="")
    else:
        query = make_mcq_prompt_without_input(question, choices, question_prefix="")
    gold = choice_labels[label_index] +". "+answer
    '''
    value = [{"role":"user","content": question}, {"role":"assistant","content": answer}]
    return value

def quartzFormatter(example, input_flag=True):
    choices = example['answer_choices']
    
    num_choices = len(example['answer_choices'])
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][:num_choices]
    question = example['inputs_pretokenized'].split('Choices:\n-')[0]
    answer = example['targets_pretokenized'].strip()
    label_index = choices.index(answer)
    if input_flag:
        query = make_mcq_prompt(question, choices)
    else:
        query = make_mcq_prompt_without_input(question, choices)
    gold = choice_labels[label_index] +". "+answer
    value = [{"role":"user","content": query}, {"role":"assistant","content": gold}]
    return value 

dataset_dict = {'cos':'cos_e_v1.11_question_description_option_text',\
    'cosmos':'cosmos_qa_context_question_description_text',\
        'dream':'dream_read_the_following_conversation_and_answer_the_question',\
            'qasc':'qasc_qa_with_separated_facts_1',\
                'quail':'quail_context_question_description_answer_text',\
                    'quartz':'quartz_paragraph_question_plain_concat',\
                        'sciq':'sciq_Multiple_Choice_Closed_Book_',\
                            'wikihop':'wiki_hop_original_choose_best_object_affirmative_1',\
                                'wiqa':'wiqa_which_of_the_following_is_the_supposed_perturbation'}
dataset_dict = {'cos':'cos_e_v1.11_question_description_option_text',\
    'cosmos':'cosmos_qa_context_question_description_text',\
        'dream':'dream_read_the_following_conversation_and_answer_the_question',\
            'qasc':'qasc_qa_with_separated_facts_1',\
                'quail':'quail_context_question_description_answer_text',\
                        'sciq':'sciq_Multiple_Choice_Closed_Book_',\
                            'wikihop':'wiki_hop_original_choose_best_object_affirmative_1',\
                                'wiqa':'allenai/wiqa'}

        #,'quarel':'quarel_heres_a_story'}
        
'''
cos: train, validation
rest all has  train, validation, test
cos: {'answer_choices': 
['house', 'autolock', 'garage', 'file cabinet', 'safe'],
'inputs': [84, 1365, 28, 6081, 3842, 4532, 2948, 58, 7023, 8, 167, 3255, 1182, 12, 1525, 8, 756, 822, 5, 17011, 10, 3, 18, 629, 3, 18, 1510, 4029, 3, 18, 3543, 3, 18, 1042, 4566, 3, 18, 1346],
'inputs_pretokenized': 'which door with lock regularly holds cars?\nChoose the most suitable option to answer the above question.\n
Options:\n- house\n- autolock\n- garage\n- file cabinet\n- safe\n',
'targets': [3543, 1],
'targets_pretokenized': '\ngarage'}
cosmos:
{'answer_choices': 
['We hung out the entire day .', 'We made sure to get lots of pictures .', 'None of the above choices .', 'We watched movies at home .'],
'inputs': [37, 207, 1506, 3, 10, 2457, 764, 12, 719, 140, 4981, 3, 5, 101, 877, 3, 27471, 3457, 13, 270, 3, 5, 37, 29, 62, 877, 12, 8, 7868, 3, 5, 37, 29, 62, 877, 3214, 300, 728, 34, 4910, 3412, 53, 3, 5, 37, 1282, 1506, 3, 10, 27, 410, 3, 29, 31, 17, 129, 1933, 3, 5, 11860, 10, 363, 2817, 227, 2457, 764, 12, 719, 3, 58, 37, 1525, 12, 8, 756, 822, 10],
'inputs_pretokenized': "The good news : Michael came to visit me yesterday . We went cruising north of here . Then we went to the gym . Then we went walking around once it stopped raining . The bad news : I did n't get pictures .\n
Question: What happened after Michael came to visit ?\n
The answer to the above question:\n",
'targets': [101, 3, 6668, 91, 8, 1297, 239, 3, 5, 1],
'targets_pretokenized': '\nWe hung out the entire day .'}
dream: 
{'answer_choices':
['She likes its color very much.', 'The price is reasonable.', 'It is on sale now.'],
'inputs': [3403, 8, 826, 3634, 11, 1525, 8, 822, 5, 549, 10, 531, 25, 114, 48, 6001, 6, 42, 1522, 27, 240, 8, 1692, 80, 58, 283, 10, 27, 114, 321, 13, 135, 6, 68, 27, 317, 8, 1692, 80, 7307, 25, 394, 5, 549, 10, 6902, 5, 27, 31, 195, 129, 8, 1692, 80, 5, 94, 31, 7, 705, 2881, 6161, 5, 11860, 10, 1615, 405, 8, 2335, 241, 12, 129, 8, 6001, 58, 3, 18, 451, 114, 7, 165, 945, 182, 231, 5, 3, 18, 37, 594, 19, 4360, 5, 3, 18, 94, 19, 30, 1048, 230, 5],
'inputs_pretokenized': "Read the following conversation and answer the question.\n\n
W: Do you like this coat, or shall I take the blue one?\n\n
M: I like both of them, but I think the blue one fits you better.\n\n
W: OK. I'll get the blue one. It's less expensive anyway.\n\n
Question: Why does the woman want to get the coat? \n\n
- She likes its color very much.\n\n
- The price is reasonable.\n\n- It is on sale now.\n",
'targets': [37, 594, 19, 4360, 5, 1],
'targets_pretokenized': '\nThe price is reasonable.'}

qasc:
{'answer_choices': 
['apples', 'trees', 'coral', 'clams', 'roses', 'wings', 'hats', 'fish'],
'inputs': [156, 2495, 33, 2303, 28, 1711, 6, 258, 125, 33, 2303, 28, 1711, 58, 11801, 3703, 10, 3, 18, 16981, 3, 18, 3124, 3, 18, 18225, 3, 18, 3, 12818, 7, 3, 18, 4659, 7, 3, 18, 13943, 3, 18, 3, 547, 7, 3, 18, 2495],
'inputs_pretokenized': 
'If fish are covered with protection, then what are covered with protection?\n\n
Answer choices:\n- apples\n - trees\n - coral\n - clams\n - roses\n - wings\n - hats\n - fish\n',
'targets': [2495, 1],
'targets_pretokenized': ' \n\n        fish  '}
quail:
{'answer_choices': 
['Steel plants will prosper', 'not enough information', 'Other businesses will prosper', 'It will keep American jobs'],
'inputs': [412, 5, 134, 5, 1661, 7459, 2523, 22, 7, 515, 12, 3, 12436, 20116, 7, 13, 944, 1093, 30, 2470, 11, 335, 1093, 30, 8749, 65, 1736, 12334, 45, 112, 8994, 66, 725, 16, 4442, 6, 186, 13, 4068, 3516, 8, 3629, 228, 7294, 3, 9, 1668, 615, 24, 8542, 412, 5, 134, 5, 1623, 5, 299, 8, 2753, 405, 43, 11172, 859, 128, 7819, 11882, 45, 2315, 213, 10861, 33, 4376, 81, 8, 307, 18, 1987, 1453, 13, 797, 3732, 2476, 5, 105, 3713, 2222, 1041, 19, 307, 147, 1259, 15, 21, 18094, 15, 26, 2470, 2677, 640, 6167, 11, 2470, 1981, 277, 113, 619, 16, 2971, 24, 70, 2476, 56, 36, 8, 416, 8926, 13, 2830, 15009, 53, 642, 13644, 451, 52, 9488, 3899, 6, 3, 9, 3, 19679, 45, 6167, 6, 243, 16, 3, 9, 2493, 1883, 227, 8, 515, 47, 2162, 5, 105, 5801, 62, 5124, 12, 1518, 95, 21, 2470, 2476, 469, 6, 1473, 56, 369, 227, 119, 2476, 95, 11, 323, 8, 1899, 3741, 5721, 1239, 797, 5347, 7021, 7, 43, 92, 23971, 3, 27433, 2523, 22, 7, 4382, 20116, 7, 6, 2145, 79, 43, 118, 24184, 21, 203, 24, 2959, 1440, 4344, 3, 31762, 70, 293, 2470, 5238, 6, 3, 3131, 797, 9216, 44, 3, 9, 18093, 5, 262, 12036, 343, 7, 43, 118, 3323, 2404, 13, 8, 515, 6, 2145, 24, 1879, 34, 56, 4781, 797, 5360, 6, 128, 13, 4068, 164, 36, 7774, 57, 3415, 3222, 21, 3, 60, 1947, 23, 6546, 26419, 5, 328, 9409, 24, 8, 1393, 12, 2470, 11, 8749, 2765, 33, 91, 24902, 57, 613, 8467, 859, 5452, 16, 119, 5238, 5, 71, 794, 13, 149, 231, 8, 962, 19, 3, 60, 739, 1014, 28, 797, 10861, 639, 416, 471, 6, 116, 10861, 16, 8913, 22, 7, 507, 189, 28167, 3939, 6, 2902, 16, 3, 9, 534, 4356, 12, 14, 3, 9, 3, 8938, 920, 3143, 5, 1404, 10861, 33, 479, 12, 8, 2753, 12, 10044, 112, 2066, 5712, 13, 9932, 3732, 2476, 16, 1371, 22, 7, 842, 40, 232, 5, 37, 1964, 21, 8, 3143, 646, 14333, 57, 7144, 5, 4485, 16212, 22, 7, 3, 7, 994, 14605, 19, 1107, 323, 12, 8, 4107, 344, 8994, 4775, 11066, 22050, 1018, 15, 11, 3, 19679, 1193, 127, 19211, 5, 22050, 1018, 15, 22, 7, 2066, 3, 24975, 2523, 22, 7, 20116, 515, 16, 3, 9, 2493, 6, 2145, 105, 5801, 119, 1440, 33, 29, 22, 17, 1556, 57, 8, 2219, 11, 20116, 7, 33, 906, 12, 1822, 2470, 11, 8749, 2476, 16, 180, 31380, 8913, 6, 233, 1615, 19, 132, 167, 952, 380, 21, 10131, 5, 2523, 31, 7, 20116, 7, 58, 8356, 8, 2024, 1525, 45, 8, 826, 931, 10, 3, 18, 6349, 2677, 56, 26282, 3, 18, 59, 631, 251, 3, 18, 2502, 1623, 56, 26282, 3, 18, 94, 56, 453, 797, 2476], 
'inputs_pretokenized': "U.S. President Donald Trump’s plan to impose tariffs of 25 percent on steel and 10 percent on aluminum has met criticism from his Republican allies in Congress, many of whom worry the measures could trigger a trade war that damages U.S. businesses.\nBut the president does have supporters among some Senate Democrats from states where voters are concerned about the long-term loss of American manufacturing jobs.\n“This welcome action is long overdue for shuttered steel plants across Ohio and steelworkers who live in fear that their jobs will be the next victims of Chinese cheating,” Senator Sherrod Brown, a Democrat from Ohio, said in a statement released after the plan was announced. “If we fail to stand up for steel jobs today, China will come after other jobs up and down the supply chain tomorrow.”\nAmerican labor unions have also broadly favored Trump’s proposed tariffs, saying they have been complaining for years that foreign countries frequently subsidize their own steel industries, putting American competitors at a disadvantage.\nEconomists have been mostly critical of the plan, saying that overall it will hurt American manufacturers, some of whom may be targeted by trading partners for retaliatory sanction. They argue that the benefits to steel and aluminum workers are outweighed by job losses among Americans in other industries.\nA test of how much the issue is resonating with American voters comes next week, when voters in Pennsylvania’s 18th congressional district, vote in a special election to fill a vacated seat.\nMany voters are looking to the president to fulfill his campaign promise of protecting manufacturing jobs in America’s heartland.\nThe race for the seat left vacant by Rep. Tim Murphy’s sex scandal is coming down to the wire between Republican candidate Rick Saccone and Democrat Conor Lamb.\nSaccone’s campaign endorsed Trump’s tariff plan in a statement, saying “If other countries aren’t playing by the rules and tariffs are needed to protect steel and aluminum jobs in Southwestern Pennsylvania,...\nWhy is there most likely support for Pres. Trump's tariffs?\nPick the correct answer from the following options:\n- Steel plants will prosper\n- not enough information\n- Other businesses will prosper\n- It will keep American jobs\n",
'targets': [59, 631, 251, 1],
'targets_pretokenized': '\nnot enough information'}
quartz:
{'answer_choices': ['more', 'less'],
'inputs': [486, 1146, 16417, 7, 6, 132, 19, 705, 799, 12, 2785, 30, 3, 9, 787, 616, 5, 282, 7798, 3, 23626, 7964, 6381, 222, 6, 132, 47, 72, 42, 705, 799, 58],
'inputs_pretokenized': 'At higher elevations, there is less air to press on a given area.\n\n
As Brian climbed Mount Everest, there was more or less air? \n\n',
'targets': [705, 1],
'targets_pretokenized': '\nless'}
sciq:
{'answer_choices': ['cell', 'marrow', 'system', 'organ'],
'inputs': [1593, 10, 363, 103, 25, 580, 3, 9, 1809, 10431, 13, 192, 42, 72, 1308, 13, 17451, 24, 161, 544, 12, 103, 3, 9, 806, 2491, 58, 13745, 7, 10, 3, 18, 2358, 3, 18, 3, 51, 6770, 3, 18, 358, 3, 18, 3640, 71, 10], 
'inputs_pretokenized': 'Q: What do you call a structure composed of two or more types of tissues that work together to do a specific task?\n\n\n Choices:\n\n- cell\n\n- marrow\n\n- system\n\n- organ\n\nA:',
'targets': [3640, 1],
'targets_pretokenized': 'organ'}
wikihop:
{'answer_choices': 
['america', 'an', 'are', 'art', 'forest', 'french', 'hamilton', 'james', 'jim', 'just', 'l .', 'lies', 'penny', 'u .', 'urban'], 
'inputs': [2784, 10, 3, 18, 1894, 41, 371, 60, 5457, 10, 3, 61, 19, 3, 9, 684, 16, 8, 8390, 985, 13, 1117, 1371, 5, 94, 7, 3, 324, 7985, 7, 11, 386, 23995, 4285, 45, 8, 9640, 12, 8, 5824, 11, 3457, 2239, 139, 8, 20132, 7087, 6, 6013, 3, 6, 492, 34, 8, 296, 31, 7, 511, 18, 15599, 7, 17, 684, 57, 792, 616, 11, 8, 4509, 18, 15599, 7, 17, 684, 57, 1322, 616, 5, 1894, 31, 7, 4947, 28, 8, 907, 1323, 19, 8, 296, 31, 7, 14783, 1322, 4947, 5, 37, 2942, 13, 8, 684, 65, 3, 9, 2107, 42, 20215, 2107, 2265, 3298, 6, 68, 3414, 49, 120, 844, 33, 1978, 16, 1248, 5, 1894, 19, 14144, 7, 15, 120, 3, 23606, 6, 8, 2942, 13, 165, 1322, 9964, 271, 3, 16032, 57, 5827, 11, 5240, 3515, 11, 8, 21989, 18183, 5, 4504, 662, 18, 89, 99, 189, 7, 13, 8, 684, 31, 7, 2074, 13, 4475, 770, 151, 19, 4150, 1601, 11, 619, 1084, 8, 7518, 4947, 5, 94, 7, 1784, 19, 19809, 6, 165, 2015, 690, 19, 7030, 117, 119, 779, 4150, 844, 560, 17219, 6, 11680, 6, 20948, 6, 26485, 6, 22649, 896, 6, 28517, 11, 13450, 5, 3, 18, 19477, 7, 13, 8, 7435, 7377, 3373, 5, 2549, 41, 683, 603, 61, 7435, 7377, 41, 683, 83, 63, 13597, 28623, 3, 18, 1600, 16047, 2038, 61, 94, 47, 6006, 7435, 7377, 7, 351, 38, 46, 21667, 3, 5525, 159, 3357, 343, 24, 3555, 376, 12, 482, 3, 9, 14978, 21, 7485, 12501, 7, 5, 86, 957, 2688, 6, 116, 6006, 7435, 7377, 47, 10725, 112, 2749, 4284, 8929, 7435, 7377, 20945, 376, 3, 9, 907, 1323, 507, 3891, 2557, 3642, 3151, 5, 3, 18, 37, 2557, 3642, 3151, 6, 92, 801, 38, 46, 2557, 3642, 23925, 6, 47, 3, 9, 80, 18, 3728, 7485, 8785, 11739, 6982, 2546, 57, 8, 907, 1323, 10504, 13, 8, 4765, 17, 45, 507, 3390, 12, 957, 12900, 94, 47, 876, 57, 2549, 1386, 17, 106, 3230, 10610, 6, 8, 5116, 695, 15299, 49, 44, 8, 9511, 4765, 17, 5, 3, 18, 2549, 262, 5, 7435, 7377, 41, 1718, 2208, 3, 6, 28623, 1636, 1600, 460, 3, 6, 2038, 3, 61, 47, 3, 9, 4151, 7485, 10543, 11, 3, 5525, 159, 4992, 14859, 3, 5, 621, 464, 38, 3, 9, 28033, 9739, 3, 6, 3, 88, 2946, 3, 9, 7485, 1078, 16, 7030, 3, 5, 216, 1790, 112, 166, 1539, 2567, 3, 6, 22424, 76, 15, 13, 4151, 15589, 7, 3, 6, 304, 2217, 7, 3, 184, 6248, 4787, 138, 31383, 3, 6, 16, 23744, 3, 5, 2502, 8342, 45, 7435, 7377, 3373, 2348, 3, 5, 3325, 7, 15, 19404, 120, 3, 88, 1916, 112, 1078, 12, 11345, 30, 112, 10142, 3, 6, 11, 865, 1916, 112, 9002, 349, 3, 6, 7435, 7377, 3373, 3, 6, 84, 3256, 12, 8099, 594, 9314, 21, 10579, 3, 6, 2137, 7977, 7, 11, 119, 2868, 2317, 7, 3, 5, 216, 2120, 910, 16, 1718, 2722, 3, 5, 3, 18, 3144, 4975, 41, 19675, 1273, 21, 96, 634, 5126, 7, 8512, 6, 8441, 8, 896, 13, 3144, 4975, 11, 557, 801, 57, 165, 2332, 7, 301, 5, 188, 5, 6, 19, 8, 511, 18, 5463, 8364, 1162, 690, 16, 8, 907, 1323, 41, 10245, 368, 1060, 896, 201, 8, 167, 8364, 1162, 690, 16, 1826, 11, 8, 5435, 3143, 13, 3144, 4975, 1334, 5, 180, 29191, 16, 5193, 1826, 6, 3144, 4975, 19, 801, 21, 165, 3, 5700, 12829, 29, 15, 152, 3298, 6, 11655, 7322, 6, 28880, 53, 12810, 15621, 6, 11, 38, 3, 9, 779, 1530, 13, 8, 797, 4527, 681, 5, 3144, 4975, 7797, 16, 3, 9, 508, 13064, 21510, 3, 8623, 30, 386, 4458, 57, 8022, 7232, 95, 12, 11, 147, 3, 5, 3, 18, 71, 7485, 19, 3, 9, 422, 6, 2667, 6, 1751, 1466, 13, 1946, 42, 2343, 261, 3, 5325, 38, 3, 9, 2768, 13, 2509, 42, 1281, 9024, 5, 328, 33, 3, 23579, 16, 1293, 6, 11, 2546, 16, 508, 16274, 44, 3, 9, 13983, 16, 455, 12, 6758, 1668, 5, 328, 33, 167, 557, 4683, 57, 3, 9, 789, 5, 3, 18, 4726, 896, 6, 8441, 896, 13, 4726, 41, 6, 3, 117, 703, 1999, 2099, 920, 38, 96, 254, 7407, 4, 8512, 6, 19, 8, 1784, 11, 167, 8364, 1162, 690, 13, 8, 907, 11511, 1323, 5, 282, 46, 96, 138, 6977, 121, 1252, 690, 6, 4726, 896, 19, 80, 13, 8, 167, 359, 981, 6881, 16, 8, 1371, 7, 5, 94, 19, 1069, 16, 8, 3460, 13, 4726, 13308, 18392, 109, 20, 10598, 226, 5807, 8512, 6, 3, 9, 508, 10645, 16, 8, 306, 20839, 7, 44, 8, 1530, 13, 4726, 6, 44, 46, 491, 6592, 13, 3, 5, 37, 690, 3, 6848, 13, 26653, 24858, 41, 2026, 19117, 120, 718, 3, 12823, 7, 137, 3, 18, 71, 1252, 690, 6, 92, 718, 296, 690, 42, 1664, 491, 6977, 690, 42, 296, 1530, 6, 19, 3, 9, 690, 2389, 1702, 12, 36, 46, 359, 150, 221, 16, 8, 1252, 1456, 358, 5, 37, 2077, 639, 45, 25055, 11, 4150, 2116, 6, 11, 8, 800, 24, 1252, 1707, 54, 36, 7571, 38, 3, 6974, 990, 6, 3, 27997, 6, 11, 3, 35, 9925, 16, 4519, 20929, 8522, 7, 1315, 12, 3, 9, 25515, 13, 3172, 12, 8, 2986, 13, 8, 1252, 358, 13, 4747, 11, 1668, 5, 3, 18, 37, 15993, 7030, 5690, 41, 517, 3221, 61, 19, 8, 167, 8364, 1162, 25233, 616, 16, 1894, 5, 486, 8, 2722, 23087, 6, 34, 141, 3, 9, 2074, 13, 8580, 3076, 8525, 2294, 4347, 11, 8, 23087, 25233, 616, 141, 3, 9, 2074, 13, 7836, 3449, 6355, 5176, 7984, 37, 15993, 7030, 5690, 19, 4802, 38, 8, 2069, 690, 13, 7030, 6, 11, 8, 662, 3518, 24858, 24, 14227, 34, 10, 25215, 6, 1626, 7377, 6, 26290, 6, 11, 1060, 5, 37, 3518, 8438, 13, 8, 15993, 7030, 5690, 19, 1664, 3334, 28, 8, 690, 13, 13450, 6, 9163, 11, 165, 3825, 1719, 6, 12, 607, 8, 15993, 7030, 11, 13450, 5690, 5, 37, 15993, 7030, 5690, 19, 8, 8390, 294, 13, 8, 7100, 13538, 5630, 15, 5, 3, 18, 3715, 41, 127, 3, 201, 8441, 8, 896, 13, 3715, 6, 19, 8, 1025, 18, 5463, 8364, 1162, 690, 16, 8, 907, 1323, 6, 11, 8, 8486, 18, 5463, 8364, 1162, 690, 16, 1117, 1371, 5, 438, 147, 3, 21280, 770, 2797, 6, 34, 19, 8, 167, 8364, 1162, 690, 16, 8, 538, 13, 7659, 11, 8, 6650, 24411, 907, 1323, 6, 11, 8, 5435, 3143, 13, 6176, 1334, 5, 37, 3715, 25233, 616, 6, 557, 3, 4822, 12, 38, 3715, 40, 232, 6, 65, 2111, 335, 770, 151, 11, 19, 8, 1025, 18, 15599, 7, 17, 16, 8, 412, 5, 134, 5, 3, 18, 7030, 19, 8, 167, 8364, 1162, 690, 16, 1894, 6, 8, 19429, 1784, 13, 9163, 6, 11, 8, 2050, 13, 8, 15993, 7030, 5690, 6, 8, 167, 8364, 1162, 25233, 616, 16, 1894, 5, 22818, 16, 2074, 6, 8, 2722, 23087, 4381, 3, 9, 2074, 13, 3547, 4241, 11116, 632, 3328, 5, 282, 13, 1230, 6, 8, 2074, 19, 230, 5861, 44, 3547, 4613, 11071, 3647, 11864, 492, 7030, 8, 4509, 18, 15599, 7, 17, 690, 16, 1117, 1371, 3, 390, 30, 8, 2074, 441, 165, 690, 6790, 5, 7030, 10583, 163, 4726, 896, 6, 368, 1060, 896, 6, 11, 3144, 4975, 57, 48, 3613, 6, 298, 34, 19, 8, 8486, 18, 15599, 7, 17, 41, 346, 2907, 26, 92, 3715, 61, 3, 99, 3, 8232, 57, 8, 812, 13, 165, 25233, 616, 3, 5, 389, 2127, 1252, 690, 6, 7030, 19, 46, 1038, 2050, 13, 268, 6, 4747, 6, 5138, 6, 11, 1543, 6, 11, 5456, 4771, 38, 80, 13, 8, 167, 29142, 11, 3, 15790, 32, 21631, 29, 3119, 16, 8, 296, 5, 3, 18, 37, 896, 13, 368, 1060, 6, 557, 718, 368, 1060, 896, 42, 914, 368, 1060, 6, 19, 8, 167, 8364, 1162, 690, 16, 8, 907, 1323, 5, 438, 46, 5861, 1230, 2074, 13, 9478, 17147, 6, 591, 3076, 8308, 147, 3, 9, 1322, 616, 13, 131, 3, 6, 368, 1060, 896, 19, 92, 8, 167, 13809, 120, 3, 23606, 779, 690, 16, 8, 907, 1323, 5, 3, 8691, 44, 8, 7518, 2226, 13, 8, 538, 13, 368, 1060, 6, 8, 690, 19, 8, 1530, 13, 8, 368, 1060, 25233, 616, 6, 80, 13, 8, 167, 8364, 1162, 4150, 3, 9, 122, 24422, 1628, 16, 8, 296, 5, 71, 1252, 579, 690, 6, 368, 1060, 896, 25914, 7, 3, 9, 1516, 1113, 1286, 12794, 6, 4747, 6, 783, 6, 768, 6, 2934, 6, 585, 6, 748, 6, 1073, 6, 11, 4527, 6, 165, 1006, 4974, 3, 16123, 8, 1657, 96, 6861, 1060, 1962, 1280, 1210, 12, 8, 13767, 13, 8, 907, 9638, 6, 368, 1060, 19, 46, 359, 1530, 21, 1038, 18320, 75, 63, 11, 65, 118, 3028, 38, 8, 2779, 11, 981, 1784, 13, 8, 296, 5, 3, 18, 9163, 6, 80, 13, 8, 1179, 7985, 7, 11, 23995, 13, 1894, 6, 19, 1069, 16, 5727, 18, 20071, 1894, 5, 94, 19, 1894, 31, 7, 167, 8364, 1162, 7985, 57, 3, 9, 508, 6346, 6, 7625, 21, 2111, 1283, 1093, 13, 66, 4151, 7, 6, 11, 19, 8, 511, 18, 15599, 7, 17, 7985, 16, 792, 616, 5, 9163, 19, 4509, 18, 15599, 7, 17, 16, 792, 616, 116, 8, 23995, 13, 8, 18194, 17495, 235, 2593, 11, 1174, 14128, 76, 17, 33, 1285, 5, 94, 19, 234, 12, 8, 2982, 31, 7, 1784, 690, 6, 19809, 6, 11, 8, 2982, 31, 7, 167, 8364, 1162, 690, 6, 7030, 5, 9246, 8, 251, 756, 6, 854, 45, 8, 570, 666, 8, 3735, 10409, 24, 6981, 7, 8, 4689, 3, 31, 17935, 29, 564, 31, 28, 8, 1426, 3, 31, 354, 603, 3, 4059, 7377, 31, 5, 13745, 7, 10, 3, 18, 3, 23064, 3, 18, 46, 3, 18, 33, 3, 18, 768, 3, 18, 5827, 3, 18, 20609, 3, 18, 3, 1483, 23, 7377, 3, 18, 7620, 15, 7, 3, 18, 3, 354, 603, 3, 18, 131, 3, 18, 3, 40, 3, 5, 3, 18, 7797, 3, 18, 23925, 3, 18, 3, 76, 3, 5, 3, 18, 4150], 
'inputs_pretokenized': 'Information:\n\n- Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world\'s second-largest country by total area and the fourth-largest country by land area. Canada\'s border with the United States is the world\'s longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country\'s population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.\n\n- Origins of the Charlton Press.\nJames (Jim) Charlton (July 26, 1911 - September 20, 2013) It was Jim Charltons experience as an avid numismatist that inspired him to create a catalogue for coin collectors. In 1926, when Jim Charlton was 15, his older brother Harry Charlton gifted him a United States 1863 Indian Head cent.\n\n- The Indian Head cent, also known as an Indian Head penny, was a one-cent coin ($0.01) produced by the United States Bureau of the Mint from 1859 to 1909. It was designed by James Barton Longacre, the Chief Engraver at the Philadelphia Mint.\n\n- James E. Charlton ( July 26 , 1911 -- September 20 , 2013 ) was a Canadian coin dealer and numismatic publisher . After working as a stationary engineer , he opened a coin store in Toronto . He published his first guidebook , Catalogue of Canadian Coins , Tokens & Fractional Currency , in 1952 . Other titles from Charlton Press followed . Subsequently he sold his store to concentrate on his publications , and later sold his publishing company , Charlton Press , which continues to publish price guides for coins , banknotes and other collectibles . He turned 100 in July 2011 .\n\n- Los Angeles (Spanish for "The Angels"), officially the City of Los Angeles and often known by its initials L.A., is the second-most populous city in the United States (after New York City), the most populous city in California and the county seat of Los Angeles County. Situated in Southern California, Los Angeles is known for its mediterranean climate, ethnic diversity, sprawling metropolis, and as a major center of the American entertainment industry. Los Angeles lies in a large coastal basin surrounded on three sides by mountains reaching up to and over .\n\n- A coin is a small, flat, round piece of metal or plastic used primarily as a medium of exchange or legal tender. They are standardized in weight, and produced in large quantities at a mint in order to facilitate trade. They are most often issued by a government.\n\n- Mexico City, officially City of Mexico (, ; abbreviated as "CDMX"), is the capital and most populous city of the United Mexican States. As an "alpha" global city, Mexico City is one of the most important financial centers in the Americas. It is located in the Valley of Mexico ("Valle de México"), a large valley in the high plateaus at the center of Mexico, at an altitude of . The city consists of sixteen municipalities (previously called boroughs).\n\n- A global city, also called world city or sometimes alpha city or world center, is a city generally considered to be an important node in the global economic system. The concept comes from geography and urban studies, and the idea that globalization can be understood as largely created, facilitated, and enacted in strategic geographic locales according to a hierarchy of importance to the operation of the global system of finance and trade.\n\n- The Greater Toronto Area (GTA) is the most populous metropolitan area in Canada. At the 2011 census, it had a population of 6,054,191, and the census metropolitan area had a population of 5,583,064. The Greater Toronto Area is defined as the central city of Toronto, and the four regional municipalities that surround it: Durham, Halton, Peel, and York. The regional span of the Greater Toronto Area is sometimes combined with the city of Hamilton, Ontario and its surrounding region, to form the Greater Toronto and Hamilton Area. The Greater Toronto Area is the northern part of the Golden Horseshoe.\n\n- Chicago (or ), officially the City of Chicago, is the third-most populous city in the United States, and the fifth-most populous city in North America. With over 2.7\xa0million residents, it is the most populous city in the state of Illinois and the Midwestern United States, and the county seat of Cook County. The Chicago metropolitan area, often referred to as Chicagoland, has nearly 10\xa0million people and is the third-largest in the U.S.\n\n- Toronto is the most populous city in Canada, the provincial capital of Ontario, and the centre of the Greater Toronto Area, the most populous metropolitan area in Canada. Growing in population, the 2011 census recorded a population of 2,615,060. As of 2015, the population is now estimated at 2,826,498, making Toronto the fourth-largest city in North America based on the population within its city limits. Toronto trails only Mexico City, New York City, and Los Angeles by this measure, while it is the fifth-largest (behind also Chicago) if ranked by the size of its metropolitan area . An established global city, Toronto is an international centre of business, finance, arts, and culture, and widely recognized as one of the most multicultural and cosmopolitan cities in the world.\n\n- The City of New York, often called New York City or simply New York, is the most populous city in the United States. With an estimated 2015 population of 8,550,405 distributed over a land area of just , New York City is also the most densely populated major city in the United States. Located at the southern tip of the state of New York, the city is the center of the New York metropolitan area, one of the most populous urban agglomerations in the world. A global power city, New York City exerts a significant impact upon commerce, finance, media, art, fashion, research, technology, education, and entertainment, its fast pace defining the term "New York minute". Home to the headquarters of the United Nations, New York is an important center for international diplomacy and has been described as the cultural and financial capital of the world.\n\n- Ontario, one of the 13 provinces and territories of Canada, is located in east-central Canada. It is Canada\'s most populous province by a large margin, accounting for nearly 40 percent of all Canadians, and is the second-largest province in total area. Ontario is fourth-largest in total area when the territories of the Northwest Territories and Nunavut are included. It is home to the nation\'s capital city, Ottawa, and the nation\'s most populous city, Toronto.\n\n\n\nGiven the information above, choose from the list below the object entity that exhibits the relation \'given name\' with the subject \'jim charlton\'.\n\nChoices:\n- america\n - an\n - are\n - art\n - forest\n - french\n - hamilton\n - james\n - jim\n - just\n - l .\n - lies\n - penny\n - u .\n - urban\n\n',
'targets': [3, 354, 603, 1],
'targets_pretokenized': '\njim'}
wiqa:
{'inputs': [10272, 10, 3, 18, 71, 686, 13, 8416, 398, 36, 3934, 274, 8, 5866, 54, 456, 3, 18, 37, 8416, 19, 4335, 323, 12, 3, 9, 2755, 1037, 3, 18, 37, 8416, 14219, 2153, 28, 1312, 799, 11, 888, 16, 12, 3, 9, 14097, 3, 18, 2336, 16, 8, 14097, 19, 3, 29496, 11, 9560, 7222, 28, 8, 8416, 3, 18, 17420, 11552, 20784, 8720, 7, 6, 84, 10224, 1427, 1567, 4107, 8416, 7, 3, 18, 37, 387, 5146, 223, 16, 12, 165, 10751, 11, 8, 433, 19, 12171, 3, 18, 11605, 8, 8416, 19, 4321, 2639, 2906, 6, 149, 56, 34, 2603, 156, 705, 8416, 19, 4335, 323, 5, 4073, 13, 8, 826, 19, 8, 5741, 29404, 257, 58, 3, 18, 1461, 1113, 53, 3, 9, 1147, 13, 8, 433, 3, 18, 25509, 1113, 53, 3, 9, 1147, 13, 8, 433, 3, 18, 59, 1113, 53, 136, 1147, 13, 8, 433],
'inputs_pretokenized': 'Process:\n\n- A type of coal must be chosen before the heating can start\n- The coal is broken down to a smaller material\n- The coal particles mix with hot air and move in to a boiler\n- Water in the boiler is pumped and produces steam with the coal\n- Steam powers turbine blades, which magnetically charge wire coals\n- The water returns back in to its chamber and the process is repeated\n- \n\nsuppose the coal is carefully selected happens, how will it affect If less coal is broken down.\n\nWhich of the following is the supposed perturbation?\n\n- directly impacting a step of the process\n- indirectly impacting a step of the process\n- not impacting any step of the process\n\n\n',
'targets': [25509, 1113, 53, 3, 9, 1147, 13, 8, 433, 1],
'targets_pretokenized': '\n\nindirectly impacting a step of the process'}
'''
drop_dataset = 'ucinlp/drop'
format_dict = {'cos':cosFormatter, 'cosmos':cosmosFormatter,'dream':dreamFormatter,'qasc':qascFormatter,\
    'quail':quailFormatter,  'sciq':sciqFormatter, 'wikihop':wikiHopFormatter,\
        'wiqa':wiqaFormatter}
index_error = defaultdict(list)
for dataset, dataset_link in dataset_dict.items():
    if dataset == 'wiqa':
        current_dataset = datasets.load_dataset('allenai/wiqa', trust_remote_code=True)
    elif dataset == 'qasc':
        current_dataset = datasets.load_dataset('allenai/qasc', trust_remote_code=True)
    else:
        current_dataset = datasets.load_dataset('bigscience/P3',dataset_link)
    print(current_dataset)

    formatter = format_dict[dataset]
    for split in ['train', 'validation', 'test']:
        if dataset in  ['wikihop', 'cos'] and split =='test':
            print(f"KeyError loading test split of dataset: {dataset}")
            break
        if dataset == 'quail' and split =='test':
            print(f"For {dataset} use challenge split instead of {split}")
            split = 'challenge'
            
        f = open(f'{input_dir}100_{dataset}_{split}.json', 'w')
        count = 0
        for data in current_dataset[split]:
            
            try:
                new_data = {'dataset':dataset, 'id':count, 'messages':formatter(data)}
            #except (IndexError, ValueError, KeyError) as e:
            except (IndexError, ValueError, KeyError) as e:
                index_error[dataset].append(data)
                print('dataset ',dataset, 'split ',split)
                continue
        
            json.dump(new_data, f)
            f.write('\n')
            count+=1
        if split == 'train':
            print('dataset ',dataset, ' ',current_dataset[split][0].keys())
#for k in index_error.keys():
#    print('k ',k, ' ',len(index_error[k]))




