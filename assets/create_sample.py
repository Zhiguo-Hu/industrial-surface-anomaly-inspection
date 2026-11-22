"""Create a deterministic synthetic fixture, not a training/evaluation sample."""
from pathlib import Path
from PIL import Image, ImageDraw

image = Image.new("RGB", (640, 480), (224, 228, 232))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((100, 70, 540, 410), radius=22, fill=(154, 164, 176), outline=(73, 84, 99), width=5)
for x, y in [(135,105), (505,105), (135,375), (505,375)]:
    draw.ellipse((x-13,y-13,x+13,y+13), fill=(70,77,88), outline=(210,215,220), width=3)
draw.line([(245,160),(273,208),(265,240),(300,275),(310,320)], fill=(50,55,62), width=4)
draw.text((18,18), "SYNTHETIC FIXTURE - PIPELINE CHECK ONLY", fill=(30,35,40))
path = Path(__file__).with_name("synthetic_inspection_sample.png")
image.save(path)
print(path)
