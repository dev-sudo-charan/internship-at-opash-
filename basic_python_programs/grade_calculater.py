print("enter your marks for the following subjects")
physics = float(input('physics : '))
chemistry = int(input('chemistry : '))
mathematics = int(input('mathematics : '))
total = ((physics+chemistry+mathematics)/300)*100
if (total > 89):
    print(" you got grade A")
elif (total > 69):
    print(" you got grade B")
elif (total > 49):
    print(" you got grade C")
elif (total > 33):
    print(" you got grade D")
else :
    print(" you have to reappear for the exam")

   