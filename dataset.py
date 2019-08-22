from random import *
import sys

def generate():

    file = open("dataset.csv", "w+")

    if file == None:
        print("File does not exist")
        sys.stderr.write()

    else:

        for i in range(1000000):
            # communication protocol
            comm_prot = randint(800,5000)

            if comm_prot >= 800 and comm_prot <= 900:
                file.write("z-wave,")
                # range of connection
                dist_range = randint(10,100)
                file.write(str(dist_range))
                file.write(',')

                # source of power
                file.write("battery,")

                # weight of device
                weight = randint(1,5)

                if weight == 1:
                    file.write("50,")

                elif weight == 2:
                    file.write("100,")

                elif weight == 3:
                    file.write("200,")

                elif weight == 4:
                    file.write("500,")

                elif weight == 5:
                    file.write("750,")

                # processing power
                proc_power = randint(0,1)

                if proc_power == 1:
                    file.write("1.3,")

                else:
                    file.write("1.0,")

                # type of device
                type = randint(0,1)

                if type == 1:
                    file.write("2")

                else:
                    file.write("1")

                file.write("\n")

            elif comm_prot >= 900 and comm_prot <= 2400:
                # communication protocol
                file.write("zigbee,")
                # range of connection
                dist_range = randint(10,100)

                file.write(str(dist_range))
                file.write(',')

                # source of power
                file.write("battery,")

                # weight of device
                weight = randint(1,5)

                if weight == 1:
                    file.write("50,")

                elif weight == 2:
                    file.write("100,")

                elif weight == 3:
                    file.write("200,")

                elif weight == 4:
                    file.write("500,")

                elif weight == 5:
                    file.write("750,")

                # processing power
                proc_power = randint(0,1)

                if proc_power == 1:
                    file.write("1.6,")

                else:
                    file.write("1.3,")

                #type of device
                type = randint(0,1)

                if type == 1:
                    file.write("2")

                else:
                    file.write("1")

                file.write("\n")

            elif comm_prot >= 2300 and comm_prot < 3000:
                # communication protocol
                prot_standard = randint(1,2)

                if prot_standard == 1:
                    file.write("bluetooth_1.0,")
                    # range of connection
                    dist_range = randint(10,100)
                    file.write(str(dist_range))
                    file.write(',')
                    # Source of power
                    file.write("battery,")
                    # weight of device
                    weight = randint(1,5)

                    if weight == 1:
                        file.write("50,")

                    elif weight == 2:
                        file.write("100,")

                    elif weight == 3:
                        file.write("200,")

                    elif weight == 4:
                        file.write("500,")

                    elif weight == 5:
                        file.write("750,")

                    # processing power
                    proc_power = randint(0,1)
                    if proc_power == 1:
                        file.write("1.6,")

                    else:
                        file.write("1.3,")

                    # type of device
                    type = randint(1,2)

                    if type == 1:
                        file.write("2")

                    else:
                        file.write("1")

                    file.write("\n")

                elif prot_standard == 2:
                    file.write("bluetooth_2.0,")
                    # range of connection
                    dist_range = randint(10,100)
                    file.write(str(dist_range))
                    file.write(',')
                    # Source of power
                    file.write("battery,")
                    # weight of device
                    weight = randint(1,5)

                    if weight == 1:
                        file.write("50,")

                    elif weight == 2:
                        file.write("100,")

                    elif weight == 3:
                        file.write("200,")

                    elif weight == 4:
                        file.write("500,")

                    elif weight == 5:
                        file.write("750,")

                    # processing power
                    proc_power = randint(0,1)

                    if proc_power == 1:
                        file.write("1.6,")

                    else:
                        file.write("1.3,")

                    # type of device
                    type = randint(0,1)

                    if type == 1:
                        file.write("2")

                    else:
                        file.write("1")

                    file.write("\n")

            elif comm_prot >= 2400 and comm_prot <= 5000:
                # communication protocol
                file.write("wi-fi,")
                # range of connection
                dist_range = randint(10,50)

                file.write(str(dist_range))
                file.write(',')

                # Source of power
                power = randint(0,1)

                if power == 1:
                    file.write("battery,")

                else:
                    file.write("ac,")

                # weight of device
                weight = randint(1,5)

                if weight == 1:
                    file.write("50,")

                elif weight == 2:
                  file.write("100,")

                elif weight == 3:
                  file.write("200,")

                elif weight == 4:
                  file.write("500,")

                elif weight == 5:
                  file.write("750,")

                # processing power
                proc_power = randint(0,1)

                if proc_power == 1:
                    file.write("0.35,")

                else:
                    file.write("0.3,")

                # type of device
                file.write("3")
                file.write("\n")

    file.close()

generate()
