from helper import Solver

if __name__ == "__main__":
  path = "images/rand.png"
  method = "lstm"

  solver = Solver()
  solver.Read_Paragraph_Image(path)
  result = solver.Recognize(method)

  print result
