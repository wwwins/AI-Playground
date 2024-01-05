from rembg import remove, new_session
from PIL import Image

# ~/.u2net
model = "u2netp"
session = new_session(model)
img = Image.open("~/Downloads/tmp-images/monster-1.png")
output = remove(img, session=session)
output.save("output.png")
