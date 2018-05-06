from helper import Solver
import tkinter as tk
from tkinter import messagebox
from tkFileDialog import askopenfilename, asksaveasfilename

def Start():
  result = solver.Recognize()
  print result


def SaveFile():
  save_path = asksaveasfilename()
  try:
    f = open(save_path, "a")
    f.write(solver.paragraph.paragraph)
    f.close()
  except:
    messagebox.showinfo("Warning", "Please specify a file")
    return 


def LoadFile():
  path = askopenfilename()
  variables['file_path'].set(path)
  try:
    solver.Read_Paragraph_Image(path)
  except:
    messagebox.showinfo("Error", "Image path error, please try again")
  window.update()


def Set_Parameters():
  if variables['file_path'].get() == "":
    messagebox.showinfo("Warning", "Please Upload Article First")
    return
  
  limit_pixel_tmp     = parameters["limit_pixel"].get("1.0",tk.END).strip()
  limit_char_tmp      = parameters["limit_char"].get("1.0",tk.END).strip()
  limit_word_mult_tmp = parameters["limit_word_mult"].get("1.0",tk.END).strip()
  target_h_tmp        = parameters["target_h"].get("1.0",tk.END).strip()
  method_tmp          = parameters["method"].get("1.0",tk.END).strip() 
  
  if limit_pixel_tmp != "":
    variables["limit_pixel"].set(limit_pixel_tmp)
  else:
    limit_pixel_tmp = variables["limit_pixel"].get()
    
  if limit_char_tmp != "":
    variables["limit_char"].set(limit_char_tmp)
  else:
    limit_char_tmp = variables["limit_char"].get()
    
  if limit_word_mult_tmp != "":
    variables["limit_word_mult"].set(limit_word_mult_tmp)
  else:
    limit_word_mult_tmp = variables["limit_word_mult"].get()

  if target_h_tmp != "":
    variables["target_h"].set(target_h_tmp)
  else:
    target_h_tmp = variables["target_h"].get()
    
  if method_tmp != "" and (method_tmp == "cnn" or method_tmp == "lstm") :
    variables["method"].set(method_tmp)
  else:
    if method_tmp != "":
      messagebox.showinfo("Warning", "Currently only support \"cnn\" and \"lstm\" methods")
    method_tmp = variables["method"].get()
  
  try:
    limit_pixel_tmp     = int(limit_pixel_tmp)
    limit_char_tmp      = int(limit_char_tmp)
    limit_word_mult_tmp = int(limit_word_mult_tmp)
    target_h_tmp        = int(target_h_tmp)
  except:
    messagebox.showinfo("Warning", "Please insert integers")
    variables["limit_pixel"]     .set("10")
    variables["limit_char"]      .set("60")
    variables["limit_word_mult"] .set("12")
    variables["target_h"]        .set("300")
    return
  
  solver.Set_Parameters(limit_pixel_tmp, limit_char_tmp, limit_word_mult_tmp, target_h_tmp, method_tmp)
  window.update()


def Layout_Setup():
  # Define Window Object
  window = tk.Tk()
  window.title('Test Window')
  window.geometry('1400x600')

  # Define Input Box
  tk.Label(window, text='Pixel\nThreshold', font=('Arial', 12),    width=10, height=2).grid(row=0,column=0,pady=5,padx=15)
  tk.Label(window, text='Word\nThreshold',  font=('Arial', 12),    width=10, height=2).grid(row=2,column=0,pady=5,padx=15)
  tk.Label(window, text='Word\nMultiplier', font=('Arial', 12),    width=10, height=2).grid(row=4,column=0,pady=5,padx=15)
  tk.Label(window, text='Target\nHeight',   font=('Arial', 12),    width=10, height=2).grid(row=6,column=0,pady=5,padx=15)
  tk.Label(window, text='Method',           font=('Arial', 12),    width=10, height=2).grid(row=8,column=0,pady=5,padx=15)

  E_limit_pixel     = tk.Text(window, width=6, height=2, font=('Arial',10) )
  E_limit_char      = tk.Text(window, width=6, height=2, font=('Arial',10) )
  E_limit_word_mult = tk.Text(window, width=6, height=2, font=('Arial',10) )
  E_target_h        = tk.Text(window, width=6, height=2, font=('Arial',10) )
  E_method          = tk.Text(window, width=6, height=2, font=('Arial',10) ) 
  
  E_limit_pixel     .grid(row=0,column=1,pady=20,padx=20)
  E_limit_char      .grid(row=2,column=1,pady=20,padx=20)
  E_limit_word_mult .grid(row=4,column=1,pady=20,padx=20)
  E_target_h        .grid(row=6,column=1,pady=20,padx=20)
  E_method          .grid(row=8,column=1,pady=20,padx=20) 
  
  limit_pixel     = tk.StringVar()
  limit_char      = tk.StringVar()
  limit_word_mult = tk.StringVar()
  target_h        = tk.StringVar()
  method          = tk.StringVar()
  
  limit_pixel     .set("10")
  limit_char      .set("60")
  limit_word_mult .set("12")
  target_h        .set("300")
  method          .set("lstm")

  tk.Label(window, textvariable = limit_pixel,      bg='gray', font=('Arial', 10),    width=8, height=2).grid(row=0,column=2,pady=5,padx=30)
  tk.Label(window, textvariable = limit_char,       bg='gray', font=('Arial', 10),    width=8, height=2).grid(row=2,column=2,pady=5,padx=30)
  tk.Label(window, textvariable = limit_word_mult,  bg='gray', font=('Arial', 10),    width=8, height=2).grid(row=4,column=2,pady=5,padx=30)
  tk.Label(window, textvariable = target_h,         bg='gray', font=('Arial', 10),    width=8, height=2).grid(row=6,column=2,pady=5,padx=30)
  tk.Label(window, textvariable = method,           bg='gray', font=('Arial', 10),    width=8, height=2).grid(row=8,column=2,pady=5,padx=30)
  
  # Define Button Objects
  file_path = tk.StringVar()
  tk.Label(window, textvariable = file_path, font=('Arial', 10), width=100, height=4, wraplength = 800).grid(row=5, rowspan=1, column=3, columnspan=3, pady=5,padx=30)

  tk.Button(window, text='Load File',        width=40, height=4,  command=LoadFile).grid(row=6,rowspan=1,column=3,columnspan=1,pady=5,padx=20)
  tk.Button(window, text='Save File',        width=40, height=4,  command=SaveFile).grid(row=6,rowspan=1,column=5,columnspan=1,pady=5,padx=20)
  tk.Button(window, text='Start',            width=40, height=4,  command=Start)   .grid(row=6,rowspan=1,column=4,columnspan=1,pady=5,padx=20)
  tk.Button(window, text='Confirm Setting',  width=20, height=4,  command=Set_Parameters) .grid(row=9,column=0,columnspan=3,pady=5,padx=20)
  
  variables = {"limit_pixel":limit_pixel, "limit_char":limit_char, "limit_word_mult":limit_word_mult,
               "target_h":target_h, "method":method, "file_path":file_path}
  
  parameters = {"limit_pixel":E_limit_pixel, "limit_char":E_limit_char, "limit_word_mult":E_limit_word_mult,
               "target_h":E_target_h, "method":E_method}
  
  return window, variables, parameters


if __name__ == "__main__":
  window, variables, parameters = Layout_Setup()
  solver = Solver()
  window.mainloop()

