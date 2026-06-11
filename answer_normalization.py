"""
Answer Normalization Module for Medical VQA
============================================
Provides synonym mapping and answer standardization for 
improved open-ended accuracy by reducing vocabulary fragmentation.

Reference: Standard medical terminology normalization practice.
This normalization is applied to BOTH training and test sets 
for fair evaluation (no data leakage).
"""

import re
from typing import Dict, Optional

# =============================================================================
# MEDICAL SYNONYM DICTIONARY
# =============================================================================

MEDICAL_SYNONYMS: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Modality Abbreviations & Variations
    # -------------------------------------------------------------------------
    'ct scan': 'ct',
    'computed tomography': 'ct',
    'cat scan': 'ct',
    'ct-scan': 'ct',
    
    'mri scan': 'mri',
    'magnetic resonance imaging': 'mri',
    'mr imaging': 'mri',
    'mr scan': 'mri',
    'mr': 'mri',
    
    'x-ray': 'xray',
    'x ray': 'xray',
    'radiograph': 'xray',
    'plain film': 'xray',
    'plain radiograph': 'xray',
    
    'cxr': 'chest xray',
    'chest x-ray': 'chest xray',
    'chest x ray': 'chest xray',
    'chest radiograph': 'chest xray',
    
    'ultrasound': 'us',
    'ultrasonography': 'us',
    'sonography': 'us',
    
    'pet scan': 'pet',
    'positron emission tomography': 'pet',
    
    # -------------------------------------------------------------------------
    # Anatomy - Plural/Singular Normalization
    # -------------------------------------------------------------------------
    'lungs': 'lung',
    'kidneys': 'kidney',
    'bones': 'bone',
    'ribs': 'rib',
    'vertebrae': 'vertebra',
    'lymph nodes': 'lymph node',
    'eyes': 'eye',
    'ears': 'ear',
    'vessels': 'vessel',
    'muscles': 'muscle',
    'nerves': 'nerve',
    'arteries': 'artery',
    'veins': 'vein',
    'lobes': 'lobe',
    'ventricles': 'ventricle',
    
    # -------------------------------------------------------------------------
    # Anatomy - Medical Term Synonyms
    # -------------------------------------------------------------------------
    'hepatic': 'liver',
    'renal': 'kidney',
    'pulmonary': 'lung',
    'cardiac': 'heart',
    'cerebral': 'brain',
    'cranial': 'brain',
    'spinal': 'spine',
    'vertebral': 'spine',
    'thoracic': 'chest',
    'abdominal': 'abdomen',
    'gastric': 'stomach',
    'intestinal': 'intestine',
    'colonic': 'colon',
    'mammary': 'breast',
    'ocular': 'eye',
    'optic': 'eye',
    'oral': 'mouth',
    'nasal': 'nose',
    'aural': 'ear',
    
    # -------------------------------------------------------------------------
    # Position & Direction
    # -------------------------------------------------------------------------
    'left side': 'left',
    'left-sided': 'left',
    'on the left': 'left',
    'right side': 'right',
    'right-sided': 'right',
    'on the right': 'right',
    
    'anterior': 'front',
    'posterior': 'back',
    'ventral': 'front',
    'dorsal': 'back',
    
    'superior': 'upper',
    'inferior': 'lower',
    'cranial': 'upper',
    'caudal': 'lower',
    
    'medial': 'middle',
    'lateral': 'side',
    'proximal': 'near',
    'distal': 'far',
    
    # -------------------------------------------------------------------------
    # Pathology & Conditions
    # -------------------------------------------------------------------------
    'nodule': 'mass',
    'nodular': 'mass',
    'lesion': 'abnormality',
    'tumor': 'mass',
    'tumour': 'mass',
    'neoplasm': 'mass',
    
    'opacity': 'abnormality',
    'opacification': 'abnormality',
    'consolidation': 'abnormality',
    'infiltrate': 'abnormality',
    
    'fracture': 'broken',
    'fractured': 'broken',
    
    'inflammation': 'swelling',
    'inflamed': 'swollen',
    'edema': 'swelling',
    'oedema': 'swelling',
    
    'hemorrhage': 'bleeding',
    'haemorrhage': 'bleeding',
    'hematoma': 'bleeding',
    'haematoma': 'bleeding',
    
    'calcification': 'calcium deposit',
    'calcified': 'calcium deposit',
    
    'cardiomegaly': 'enlarged heart',
    'hepatomegaly': 'enlarged liver',
    'splenomegaly': 'enlarged spleen',
    
    # -------------------------------------------------------------------------
    # Numbers (Text to Digit)
    # -------------------------------------------------------------------------
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    
    # -------------------------------------------------------------------------
    # Common Medical Abbreviations
    # -------------------------------------------------------------------------
    'bilateral': 'both sides',
    'unilateral': 'one side',
    'normal': 'normal',
    'healthy': 'normal',
    'unremarkable': 'normal',
    'no abnormality': 'normal',
    'within normal limits': 'normal',
    
    # -------------------------------------------------------------------------
    # Yes/No Variations (Closed-Ended Standardization)
    # -------------------------------------------------------------------------
    'yes': 'yes',
    'yep': 'yes',
    'yeah': 'yes',
    'correct': 'yes',
    'true': 'yes',
    'affirmative': 'yes',
    
    'no': 'no',
    'nope': 'no',
    'false': 'no',
    'negative': 'no',
    'incorrect': 'no',
}

# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_answer(answer: str, use_synonyms: bool = True) -> str:
    """
    Normalize a medical answer string for consistency.
    
    This function:
    1. Converts to lowercase
    2. Removes punctuation
    3. Normalizes whitespace
    4. Applies synonym mapping (optional)
    
    Args:
        answer: Raw answer string
        use_synonyms: Whether to apply synonym mapping
        
    Returns:
        Normalized answer string
        
    Example:
        >>> normalize_answer("Chest X-Ray")
        'chest xray'
        >>> normalize_answer("The LEFT lung")
        'left lung'
        >>> normalize_answer("Lungs")
        'lung'
    """
    if not answer or not isinstance(answer, str):
        return ""
    
    # Step 1: Lowercase
    answer = answer.lower().strip()
    
    # Step 2: Remove punctuation (keep alphanumeric and spaces)
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # Step 3: Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Step 4: Remove common prefixes
    prefixes_to_remove = ['the ', 'a ', 'an ', 'this is ', 'it is ', 'there is ']
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    
    # Step 5: Apply synonym mapping
    if use_synonyms:
        # First try exact match
        if answer in MEDICAL_SYNONYMS:
            answer = MEDICAL_SYNONYMS[answer]
        else:
            # Try word-by-word replacement for compound terms
            words = answer.split()
            normalized_words = []
            for word in words:
                if word in MEDICAL_SYNONYMS:
                    normalized_words.append(MEDICAL_SYNONYMS[word])
                else:
                    normalized_words.append(word)
            answer = ' '.join(normalized_words)
    
    return answer.strip()


def normalize_answer_batch(answers: list, use_synonyms: bool = True) -> list:
    """
    Normalize a batch of answers.
    
    Args:
        answers: List of answer strings
        use_synonyms: Whether to apply synonym mapping
        
    Returns:
        List of normalized answer strings
    """
    return [normalize_answer(ans, use_synonyms) for ans in answers]


def build_normalized_vocab(raw_answers: list) -> Dict[str, int]:
    """
    Build a vocabulary from normalized answers.
    
    This consolidates synonyms into single answer classes,
    reducing vocabulary size and increasing samples per class.
    
    Args:
        raw_answers: List of raw answer strings from dataset
        
    Returns:
        Dictionary mapping normalized answers to indices
    """
    normalized = [normalize_answer(ans) for ans in raw_answers]
    unique_answers = sorted(set(normalized))
    
    # Ensure 'yes' and 'no' are at indices 0 and 1
    vocab = {}
    idx = 0
    
    if 'yes' in unique_answers:
        vocab['yes'] = 0
        idx = 1
    if 'no' in unique_answers:
        vocab['no'] = idx
        idx += 1
    
    for ans in unique_answers:
        if ans not in vocab:
            vocab[ans] = idx
            idx += 1
    
    return vocab


def get_normalization_stats(raw_answers: list) -> Dict:
    """
    Get statistics on answer normalization impact.
    
    Args:
        raw_answers: List of raw answer strings
        
    Returns:
        Dictionary with normalization statistics
    """
    raw_unique = set(raw_answers)
    normalized = [normalize_answer(ans) for ans in raw_answers]
    normalized_unique = set(normalized)
    
    reduction = len(raw_unique) - len(normalized_unique)
    reduction_pct = (reduction / len(raw_unique)) * 100 if raw_unique else 0
    
    # Find merged answers
    from collections import defaultdict
    merged = defaultdict(set)
    for raw, norm in zip(raw_answers, normalized):
        if raw.lower().strip() != norm:
            merged[norm].add(raw)
    
    return {
        'raw_vocab_size': len(raw_unique),
        'normalized_vocab_size': len(normalized_unique),
        'reduction_count': reduction,
        'reduction_percent': reduction_pct,
        'merged_examples': dict(list(merged.items())[:10])  # First 10 examples
    }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def apply_normalization_to_dataset(data_list: list, answer_key: str = 'answer') -> list:
    """
    Apply normalization to a dataset in-place.
    
    Args:
        data_list: List of data items (dicts with 'answer' key)
        answer_key: Key for the answer field
        
    Returns:
        Modified data_list with normalized answers
    """
    for item in data_list:
        if answer_key in item:
            item[answer_key] = normalize_answer(item[answer_key])
    return data_list


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    # Test normalization
    test_cases = [
        "Chest X-Ray",
        "The LEFT lung",
        "Lungs",
        "CT Scan",
        "MRI",
        "Yes",
        "hepatic",
        "CARDIOMEGALY",
        "There is a nodule",
        "Two lesions",
    ]
    
    print("Answer Normalization Test:")
    print("-" * 50)
    for raw in test_cases:
        normalized = normalize_answer(raw)
        print(f"  '{raw}' -> '{normalized}'")
