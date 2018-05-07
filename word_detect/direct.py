from helper import Solver
  
path = "images/rand.png"
limit_pixel = 10
limit_char = 34
limit_word_mult = 12
target_h = 300
method = "lstm"

solver = Solver()
solver.Read_Paragraph_Image(path)
solver.Set_Parameters(limit_pixel, limit_char, limit_word_mult, target_h, method)
result = solver.Recognize()
print result
