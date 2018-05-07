from helper import Solver
  
path = "images/test_paragraph3.png"
solver = Solver()
solver.Read_Paragraph_Image(path)
result = solver.Recognize()
print result
