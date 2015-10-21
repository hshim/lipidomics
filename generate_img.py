
def create_imgs(lipid_names):
  from PIL import Image
  from PIL import ImageFont
  from PIL import ImageDraw 
  i = 0
  for lipid_name in lipid_names:
    img = Image.open("blank.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 75)
    draw.text((0, 0),lipid_name,(255,255,255),font=font)
    img.save(str(i) + '.png')
    i += 1


