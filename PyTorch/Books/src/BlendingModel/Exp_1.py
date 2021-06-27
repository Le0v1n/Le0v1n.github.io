seasons = ['Spring', 'Summer', 'Fall', 'Winter']

for idx, data in list(enumerate(seasons)):
    print("idx: {}, data: {}    ".format(idx, data), end="")
    # Res: idx: 0, data: Spring    idx: 1, data: Summer    idx: 2, data: Fall    idx: 3, data: Winter
print("")

for idx, data in list(enumerate(seasons, 0)):
    print("idx: {}, data: {}    ".format(idx, data), end="")
    # Res: idx: 0, data: Spring    idx: 1, data: Summer    idx: 2, data: Fall    idx: 3, data: Winter
print("")

for idx, data in list(enumerate(seasons, start=1)):
    print("idx: {}, data: {}    ".format(idx, data), end="")
    # Res: idx: 1, data: Spring    idx: 2, data: Summer    idx: 3, data: Fall    idx: 4, data: Winter
print("")

for idx, data in list(enumerate(seasons, start=5)):
    print("idx: {}, data: {}    ".format(idx, data), end="")
    # Res: idx: 5, data: Spring    idx: 6, data: Summer    idx: 7, data: Fall    idx: 8, data: Winter
