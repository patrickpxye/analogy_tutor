import random
import string
import re

# generate a hashmap for concepts to random strings, and record the mapping
def generate_anonymized_mapping(concepts):

    concepts = list(set([concept.lower() for concept in concepts]))

    def random_string(length=8):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    mapping = {}
    for concept in concepts:
        anonymized = random_string()
        while anonymized in mapping.values():
            anonymized = random_string()
        mapping[concept] = anonymized
    
    return mapping

# anonymize the concepts in a text using the mapping, accounting for plural forms, capitalization, and partial matches
def anonymize_text(text, mapping):
    # Sort concepts by length in descending order to prioritize multi-word concepts
    sorted_concepts = sorted(mapping.keys(), key=len, reverse=True)
    
    # Escape special regex characters in concepts to avoid unintended behavior
    escaped_concepts = [re.escape(concept) for concept in sorted_concepts]
    
    # Build a regex pattern that matches any concept as a whole word
    pattern = r'\b(' + '|'.join(escaped_concepts) + r')\b'
    
    def replace_match(match):
        word = match.group(0)
        lower_word = word.lower()
        
        # Look up the lower-cased version in the mapping
        if lower_word in mapping:
            replacement = mapping[lower_word]
        elif lower_word.endswith('s') and lower_word[:-1] in mapping:  # plural form
            replacement = mapping[lower_word[:-1]] + 's'
        else:
            return word  # No replacement if word not in mapping
        
        # Preserve original capitalization
        if word.istitle():
            return replacement.capitalize()
        elif word.isupper():
            return replacement.upper()
        else:
            return replacement
    
    # Apply replacements using the regex pattern
    anonymized_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    
    return anonymized_text