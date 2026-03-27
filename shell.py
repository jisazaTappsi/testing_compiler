import basic

while True:
    text = input('basic > ')
    result, error, _ = basic.run_ai('<stdin>', text)
    if error: print(error.as_string())
    elif result: print(result)
