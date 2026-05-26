def safe_counter(token, store=None):

    if store is None:
        store = {}

    if token in store:
        store[token] += 1
    else:
        store[token] = 1

    return store





if __name__=="__main__":
    counter1 = safe_counter("spam")
    counter2 = safe_counter("spam")
    print(counter1)
    print(counter2)