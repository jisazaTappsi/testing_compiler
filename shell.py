import basic

while True:
    text = input('basic > ')
    result, error = basic.run_ai('<stdin>', text)
    if error: print(error.as_string())
    else: print(result)
