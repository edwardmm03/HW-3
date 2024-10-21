if __name__ == "__main__":
    test = [(1,2),(2,4),(3,6),(4,8),(5,10)]
    m = len(test)
    distribution = []

    #setting the initial distribution
    for i in test:
        distribution.append(1/m)

    print(distribution[1])
    print(distribution[m-1])

    #adaboost algorithm
    for x in 5:
        ht #classifier
        at = calcalpha(error) #calculate alphat
        zt = 2*pow(error*(1-error), .5) #calculate normalization factor
        for i in m:
            #update the distribution
            distribution[i] = (pow(distribution[i], -at*yi*ht(xi)))/zt
        f = atht #final hypothesis H
