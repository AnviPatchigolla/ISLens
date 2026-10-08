"""Single source of truth for the phrases this prototype recognizes.

Both the recorder tool and the extension fetch this list from GET /phrases
so every part of the system agrees on the label set and ordering.
"""

PHRASES = [
    "hello",
    "good_morning",
    "help",
    "thank_you",
    "stop",
]

PHRASE_DISPLAY = {
    "hello": "Hello",
    "good_morning": "Good Morning",
    "help": "Help",
    "thank_you": "Thank You",
    "stop": "Stop",
}

PHRASE_DISPLAY_HI = {
    "hello": "नमस्ते",
    "good_morning": "सुप्रभात",
    "help": "मदद",
    "thank_you": "धन्यवाद",
    "stop": "रुको",
}
