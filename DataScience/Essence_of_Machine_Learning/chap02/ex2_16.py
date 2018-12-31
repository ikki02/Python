with open("output.txt", "w") as fw:
    with open("sample.txt") as fr:
        for line in fr:
            print(line, end="", file=fw)
