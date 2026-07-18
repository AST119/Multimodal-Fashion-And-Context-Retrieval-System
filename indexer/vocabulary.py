# Colab-pasteable: yes
"""
Module defining the structured visual vocabulary categories and 
the global context axes used for scene-level semantic indexing.
"""

VOCABULARY = [
    # Upperbody
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "blazer", "hoodie", "suit",
    # Lowerbody
    "pants, trousers, jeans", "shorts", "skirt", 
    # Wholebody
    "coat", "dress", "one-piece, romper", "two-piece, co-ord set", "evening gown, formal dress", "wedding dress", "jumpsuit", "cape", 
    # Accessories & Footwear
    "hat", "glasses, sunglasses", "tie", "glove", "watch", "belt", "sock", "shoe", "bag, handbag", "scarf", "umbrella"
]

CONTEXT_AXES = {
    "places": [
        "office interior, indoor workspace", 
        "urban street, outdoor city", 
        "park, garden, outdoor nature", 
        "home interior, indoor living room",
        "beach, outdoor coast",
        "indoor cafe, restaurant"
    ],
    "seasons": [
        "sunny summer weather", 
        "cold winter weather", 
        "spring season", 
        "autumn fall season"
    ],
    "vibes": [
        "formal business, professional style", 
        "casual weekend, relaxed style", 
        "sporty, athletic active style", 
        "glamorous, elegant luxury style"
    ],
    "actions": [
        "stand", 
        "sit", 
        "walk", 
        "run",
        "pose"
    ]
}