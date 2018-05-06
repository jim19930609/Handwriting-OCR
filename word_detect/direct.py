from helper import Solver
  
path = "images/rand.png"
solver = Solver()
solver.Read_Paragraph_Image(path)
result = solver.Recognize()
print result
