from matplotlib import pyplot as plt
import numpy as np 
import random
try : 
    x
    years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 400.6, 14958.3]

    # create a line chart, years on x-axis, gdp on y-axis
    plt.plot(years, gdp, color='green', marker='x', linestyle='solid')

    # add a title
    plt.title("Nominal GDP")

    # add a label to the y-axis
    plt.ylabel("Billions of $")
    plt.xlabel("MONEY")
    plt.show()

    # label x-axis with movie names at bar centers
    plt.bar(range(len(years)), gdp)
    plt.xticks(range(len(years)), years)

    plt.show()
    from collections import Counter
    grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

    # Bucket grades by decile, but put 100 in with the 90s
    histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
    print(histogram)

    plt.bar([x + 5 for x in histogram.keys()],  # Shift bars right by 5
            histogram.values(),                 # Give each bar its correct height
            10,                                 # Give each bar a width of 10
            edgecolor=(0, 0, 0))                # Black edges for each bar

    plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105,
                                            # y-axis from 0 to 5

    plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
    plt.xlabel("Decile")
    plt.ylabel("# of Students")
    plt.title("Distribution of Exam 1 Grades")
    plt.show()

    # #line graph
# plt.plot(x,y,marker='o', color = 'red',linestyle = 'solid')
# plt.title('this is the code')
# plt.show()

# #bar graph
# plt.bar(x,y) 
# plt.xticks(x,y) #shows values for all the raise in bars 
# plt.show()

    x = [500,505]

    y = [2017,2018]
    print(x,y)

    plt.bar(y,x,0.8) 
    plt.xticks(y)
    # plt.ticklabel_format(useoffset=False) 
    plt.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(useOffset=False)
    plt.axis([2016.5,2018.5,495,506])
    plt.show()

    plt.axis([2016.5,2018.5,0,506])
    plt.show()
    # mentions = [500, 505]
    # years = [2017, 2018]

    # plt.bar(years, mentions, 0.8)
    # plt.xticks(years)
    # plt.ylabel("# of times I heard someone say 'data science'")

    # # if you don't do this, matplotlib will label the x-axis 0, 1
    # # and then add a +2.013e3 off in the corner (bad matplotlib!)
    # plt.ticklabel_format(useOffset=False)

    # # misleading y-axis only shows the part above 500
    # plt.axis([2016.5, 2018.5, 499, 506])
    # plt.title("Look at the 'Huge' Increase!")
    # plt.show()


except:
    a = 1

# v= [1, 2, 4, 8, 16, 32, 64, 128, 256]
# b = [256, 128, 64, 32, 16, 8, 4, 2, 1]
# total_error = [x + y for x, y in zip(v,b)]
# print(total_error)
# xs =[i for i, _ in enumerate(v)]
# print(xs)
# plt.plot(xs, v,'g-', label='v')
# plt.plot(xs, b, 'r-.', label='bias^2')
# plt.plot(xs, total_error,'b:',label='total error')
# plt.legend(loc=5)
# plt.xlabel("model complexity")
# plt.xticks([])
# plt.title("The Bias-v Tradeoff")
# plt.show()


x= random.sample(range(1, 100), 10)
print(x)
y= random.sample(range(1, 100), 10)
y.reverse()
print(y)
totalerror = [u+v for u,v in zip(x,y)]
print(totalerror)
xs = [ i for i,_ in enumerate(x)]

plt.plot(xs,x,'g-',label='xxx')
plt.plot(xs,y,'b-.',label = 'yyy')
plt.plot(xs,totalerror,'r:',label = 'error')
plt.legend(loc=9)
plt.xticks([])
plt.show()