from typing import List
import re

def simplify_sentence(sentence):
    """Removes all characters expect A-Za-z0-9 and whitespaces

    Args:
        sentence (str): Input sentence

    Returns:
        str: Output sentence
    """
    # return sentence.replace('.','').replace(',','').replace(';','').lower()
    return re.sub(r'[^A-Za-z0-9 ]+', '', sentence)
