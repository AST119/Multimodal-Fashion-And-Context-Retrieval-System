# Colab-pasteable: yes
"""
Configuration module containing local/remote directory paths, model constants, 
filtering thresholds, and the hierarchical Fashionpedia attributes mapping.
"""
import os

# Centralized Directory Paths
IMAGE_DIR = "./data/images"
OUTPUT_DIR = "./vector_store/output"
CROPS_DIR = "./crops"

# Pre-trained Model Identifiers
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"

# Filtering Thresholds
CLIP_PREFILTER_THRESHOLD = 0.22
CLIP_PREFILTER_TOP_K = 5
CLIPSEG_CONF_THRESHOLD = 0.5

# Hierarchical Fashionpedia attributes focusing on pattern, material, and length
FASHION_ATTRIBUTES_MAP = {
    "shirt, blouse": {
        "pattern": ["plain", "striped", "plaid", "check", "polka dot", "floral", "graphic print", "paisley", "abstract"],
        "material": ["cotton", "silk", "linen", "satin", "flannel", "lace", "chiffon"],
        "neckline": ["v-neck", "crew neck", "round neck", "scoop neck", "boat neck", "high neck", "collarless", "off-the-shoulder"]
    },
    "top, t-shirt, sweatshirt": {
        "pattern": ["plain", "striped", "graphic print", "tie-dye", "letters, numbers", "cartoon"],
        "material": ["cotton", "polyester", "fleece", "jersey"],
        "nickname": ["crop", "halter", "camisole", "tank", "peasant", "tube", "tunic", "hoodie"]
    },
    "sweater": {
        "pattern": ["plain", "striped", "fair isle", "cable knit", "argyle", "herringbone"],
        "material": ["wool", "cashmere", "acrylic", "cotton", "mohair", "distressed"],
        "neckline": ["turtleneck", "v-neck", "crew neck", "cowl neck", "mock neck", "turtle"]
    },
    "cardigan": {
        "pattern": ["plain", "striped", "check", "ribbed knit"],
        "material": ["wool", "cashmere", "cotton", "chenille"]
    },
    "jacket": {
        "pattern": ["plain", "check", "striped", "camouflage", "leopard"],
        "material": ["leather", "suede", "denim", "nylon", "wool", "puffer", "fur", "shearling", "distressed"],
        "length": ["cropped", "standard length", "longline"],
        "nickname": ["blazer", "puffer jacket", "biker jacket", "bomber jacket", "windbreaker", "varsity jacket", "trucker jacket", "utility jacket"]
    },
    "blazer": {
        "pattern": ["plain", "check", "pinstripe", "plaid", "houndstooth"],
        "material": ["wool", "linen", "velvet", "polyester", "corduroy"]
    },
    "coat": {
        "pattern": ["plain", "check", "houndstooth", "herringbone", "chevron"],
        "material": ["wool", "fur", "shearling", "cashmere", "waterproof nylon", "puffer", "leather"],
        "length": ["knee length", "midi", "maxi length", "duster"],
        "nickname": ["trench coat", "parka", "pea coat", "shearling coat", "teddy bear coat", "puffer coat", "raincoat", "duffle coat"]
    },
    "pants, trousers, jeans": {
        "pattern": ["plain", "camouflage", "check", "striped"],
        "material": ["denim", "leather", "cotton", "fleece", "corduroy", "velvet", "washed"],
        "silhouette": ["baggy", "wide leg", "straight", "tight fit", "bell bottom", "bootcut", "skinny", "cargo"]
    },
    "shorts": {
        "pattern": ["plain", "check", "striped", "floral"],
        "material": ["denim", "cotton", "polyester", "linen"]
    },
    "skirt": {
        "pattern": ["plain", "floral", "plaid", "striped", "dot", "geometric"],
        "material": ["denim", "leather", "silk", "pleated polyester", "tulle", "satin"],
        "length": ["mini length", "midi", "maxi length"]
    },
    "dress": {
        "pattern": ["plain", "floral", "geometric", "paisley", "striped", "polka dot", "animal print", "abstract"],
        "material": ["silk", "satin", "chiffon", "velvet", "lace", "tulle", "cotton", "knit", "organza"],
        "length": ["mini length", "midi", "maxi length", "floor length"],
        "nickname": ["slip dress", "wrap dress", "bodycon dress", "gown", "sundress", "kaftan", "shirt dress", "skater dress"]
    },
    "two-piece, co-ord set, matching set": {
        "pattern": ["plain", "striped", "check", "floral", "printed"],
        "material": ["linen", "knitwear", "cotton", "silk"]
    },
    "evening gown, formal dress, cocktail dress": {
        "pattern": ["plain", "sequined", "beaded", "embroidered", "metallic"],
        "material": ["luxury silk", "chiffon", "velvet", "satin", "lace", "tulle", "organza"],
        "length": ["maxi length", "floor length", "high-low"]
    },
    "wedding dress, bridal gown": {
        "pattern": ["plain", "lace patterned", "embroidered", "beaded"],
        "material": ["luxury lace", "satin", "tulle", "organza", "chiffon"],
        "length": ["floor length", "cathedral train"]
    },
    "bag, handbag, tote": {
        "pattern": ["plain", "monogram", "quilted", "crocodile texture", "snakeskin"],
        "material": ["leather", "canvas", "straw woven", "suede", "nylon"]
    }
}