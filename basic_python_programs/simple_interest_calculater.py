principal_amount = int(input("enter  principal amount : "))
rate_of_interest = float(input("enter interest rate : "))
time_taken = int(input("enter time taken (in months): "))
simple_interest = (principal_amount * rate_of_interest * time_taken)/100
print("simple_interest is :" ,simple_interest)