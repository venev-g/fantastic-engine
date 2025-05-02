## Overview

The NLU module uses spaCy to perform two main tasks:
1. Intent classification - detecting the purpose of a user's message
2. Entity extraction - identifying important information pieces in text

## Dependencies

- [spaCy](https://spacy.io/) library
- English language model (`en_core_web_sm`)

## Main Components

### 1. Intent Classifier

The `classify_intent` function determines the user's intent by matching keywords in the input text against predefined patterns.

Supported intents:
- **greeting**: hello, hi, hey, greetings
- **goodbye**: bye, goodbye, see you, farewell
- **booking**: book, reserve, schedule
- **weather**: weather, forecast, temperature
- **order_food**: order, food, meal, dinner, buy, get
- **unknown**: default when no patterns match

### 2. Entity Extractor

The `extract_entities` function identifies and extracts named entities and numeric values from the input text.

Entity types extracted:
- Named entities recognized by spaCy (locations, dates, organizations, etc.)
- Numeric values (marked as 'number')

### 3. Sentence Analyzer

The `analyze_sentence` function combines intent classification and entity extraction to provide complete analysis of user input.

Returns:
- The classified intent
- A dictionary of extracted entities with their types

## Usage Example

```python
text = "Book a flight to Paris next Monday"
text_lower = text.lower()
result = analyze_sentence(text_lower)

# Result:
# Intent: booking
# Entities: {'GPE': 'paris', 'DATE': 'next monday'}
```

## Processing Flow

1. Text is lowercased for consistent processing
2. Intent is classified based on keyword matching
3. Entities are extracted using spaCy's NLP capabilities
4. Results are combined into a single analysis output

## Sample Results

| Input Text | Detected Intent | Extracted Entities |
|------------|----------------|-------------------|
| "Hello, I would like to book a table for two." | greeting | {'CARDINAL': 'two', 'number': 'two'} |
| "What's the weather like today?" | weather | {'DATE': 'today'} |
| "Book a flight to Paris next Monday." | booking | {'GPE': 'paris', 'DATE': 'next monday'} |
