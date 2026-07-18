"""
Color Bucket Map Module.
Maps specific CSS3 color names to 14 standard parent color buckets.
Fully compatible with webcolors v25.10.0+ public API.
"""
import webcolors

BASIC_BUCKETS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", 
    "brown", "black", "white", "gray", "beige", "navy", "olive"
]

COLOR_BUCKETS = {
    "red": (255, 0, 0), "orange": (255, 127, 0), "yellow": (255, 255, 0),
    "green": (0, 255, 0), "blue": (0, 0, 255), "purple": (127, 0, 127),
    "pink": (255, 192, 203), "brown": (150, 75, 0), "black": (0, 0, 0),
    "white": (255, 255, 255), "gray": (128, 128, 128), "beige": (245, 245, 220),
    "navy": (0, 0, 128), "olive": (128, 128, 0)
}

COLOR_BUCKET_MAP = {}

try:
    # Retrieve all valid names and their RGB values using the public API
    css3_names = webcolors.names(spec='css3')
    for name in css3_names:
        rgb = webcolors.name_to_rgb(name, spec='css3')
        
        # Match each CSS3 name to the closest basic bucket based on Euclidean distance
        best_bucket = "gray"
        min_dist = float('inf')
        for b_name, b_rgb in COLOR_BUCKETS.items():
            dist = (rgb.red - b_rgb[0])**2 + (rgb.green - b_rgb[1])**2 + (rgb.blue - b_rgb[2])**2
            if dist < min_dist:
                min_dist = dist
                best_bucket = b_name
        COLOR_BUCKET_MAP[name] = best_bucket
        
except Exception as e:
    print(f"[Warning] Failed to dynamically map color buckets via webcolors ({str(e)}). Using basic fallback.")
    COLOR_BUCKET_MAP = {color: color for color in BASIC_BUCKETS}