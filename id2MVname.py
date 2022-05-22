def getMvDict():
    file_object = open("ml-100k/u.item",'rb')
    MVdic ={}
    i=0
    try:
        for line in file_object:
            i+=1
            #print(line)
            array = line.split("|".encode())
            name =(str(array[1])[2:-1])
            #print(name)
            MVdic[i]=name
    finally:
         file_object.close()
         return MVdic