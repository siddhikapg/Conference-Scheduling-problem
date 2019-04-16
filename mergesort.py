def mergeIt(array, start, mid, end):
    # create a temporary array
    temp = list()

    # Assign input to temporary variables as we need to alter them in the process
    i = start;
    m = mid;
    j = mid + 1;  # This is how we logically divide array in 2 parts
    index = 0;  # This is index to mark progress of temp array

    # compare the elements of 2 logical parts of array and
    # fill the temp array accordingly
    while i <= m and j <= end:
        if (array[i] <= array[j]):
            temp.append(array[i]);
            # index+= 1
            i += 1
        else:
            temp.append(array[j]);
            # index += 1
            j += 1

    # The cool thing is that, after the while loop ends, we will have some indexes from both
    # logical halfs and we put them in rest of the temp array

    while (i <= m):
        temp.append(array[i]);
        # index += 1
        i += 1

    while (j <= end):
        temp.append(array[j]);
        # index += 1
        j += 1

    # copy the temp array in main array
    index = 0
    for i in range(start, end + 1):
        array[i] = temp[index]
        index += 1
    # for(i =start,;i <= end;++i, ++index)
    #   array[i] = temp[index];


def mergeLoop(array, start, end):
    # This method calls itself recursively
    # till it breaks down input array into arrays of size 1
    if (start < end):
        # calculate mid point
        # Mathematically right shift of a value is value/2
        mid = (start + end) >> 1
        mergeLoop(array, start, mid)
        mergeLoop(array, mid + 1, end)
        mergeIt(array, start, mid, end)


def mergeSort(array, size):
    mergeLoop(array, 0, size - 1)

