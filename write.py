import os

def change_property(file, etas, etap, lamdaD, lamdaR, chimax, beta, delta, lamda):
    with open(file, "rb") as f1, open("%s.bak" % file, "wb") as f2:
        i = 0
        for line in f1:
            i += 1
            if i == 30:
                line = b"                etaS             etaS [1 -1 -1 0 0 0 0] " + str.encode(str(etas)) + b";\n"
            if i == 31:
                line = b"                etaP             etaP [1 -1 -1 0 0 0 0] " + str.encode(str(etap)) + b";\n"
            if i == 32:
                line = b"                lambdaD          lambdaD [0 0 1 0 0 0 0] " + str.encode(str(lamdaD)) + b";\n"
            if i == 33:
                line = b"                lambdaR          lambdaR [0 0 1 0 0 0 0] " + str.encode(str(lamdaR)) + b";\n"
            if i == 34:
                line = b"                chiMax           chiMax [0 0 0 0 0 0 0] " + str.encode(str(chimax)) + b";\n"
            if i == 35:
                line = b"                beta             beta [0 0 0 0 0 0 0] " + str.encode(str(beta)) + b";\n"
            if i == 36:
                line = b"                delta            delta [0 0 0 0 0 0 0] " + str.encode(str(delta)) + b";\n"
            if i == 37:
                line = b"	lambda           lambda [0 0 1 0 0 0 0] " + str.encode(str(lamda)) + b";\n"
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


def write_property(file, order, etas, etap, lamdaD, lamdaR, chimax, beta, delta, lamda):
    with open(file, "rb") as f1, open("./property/"+str(order), "wb") as f2:
        i = 0
        for line in f1:
            i += 1
            if i == 30:
                line = b"                etaS             etaS [1 -1 -1 0 0 0 0] " + str.encode(str(etas)) + b";\n"
            if i == 31:
                line = b"                etaP             etaP [1 -1 -1 0 0 0 0] " + str.encode(str(etap)) + b";\n"
            if i == 32:
                line = b"                lambdaD          lambdaD [0 0 1 0 0 0 0] " + str.encode(str(lamdaD)) + b";\n"
            if i == 33:
                line = b"                lambdaR          lambdaR [0 0 1 0 0 0 0] " + str.encode(str(lamdaR)) + b";\n"
            if i == 34:
                line = b"                chiMax           chiMax [0 0 0 0 0 0 0] " + str.encode(str(chimax)) + b";\n"
            if i == 35:
                line = b"                beta             beta [0 0 0 0 0 0 0] " + str.encode(str(beta)) + b";\n"
            if i == 36:
                line = b"                delta            delta [0 0 0 0 0 0 0] " + str.encode(str(delta)) + b";\n"
            if i == 37:
                line = b"	lambda           lambda [0 0 1 0 0 0 0] " + str.encode(str(lamda)) + b";\n"
            f2.write(line)
    # os.remove(file)
    # os.rename("%s.bak" % file, file)