from matplotlib.ticker import MaxNLocator
import itertools

def draw_rectangles(rectangleDataList):
    padding = 5
    maxX = max([max_x for (min_x,min_y,max_x,max_y) in rectangleDataList]) + padding
    minX = min([min_x for (min_x,min_y,max_x,max_y) in rectangleDataList]) - padding
    maxY = max([max_y for (min_x,min_y,max_x,max_y) in rectangleDataList]) + padding
    minY = min([min_y for (min_x,min_y,max_x,max_y) in rectangleDataList]) - padding

    # this plots all of the rectangles on the same graph
    fig, ax = plt.subplots()
    plt.title('Rectangles')
    ax.set_aspect(1) #normalizes the graph
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #the x and y limits for the graph are set by the largest x and y values generated and are restricted by radii
    plt.xlim(minX , maxX )
    plt.ylim(minY , maxY )
    plt.grid(True, which='both')

    for rectangle in rectangleDataList:
        min_x,min_y,max_x,max_y = rectangle
        width = max_x - min_x
        height = max_y - min_y

        # Plot library uses min left as the point
        pt_x = min_x
        pt_y = min_y

        # For annotation
        center_x = min_x + width/2.0
        center_y = min_y + height/2.0

        # Place the rectangle
        rectangleObj = plt.Rectangle(xy=(pt_x, pt_y), width=width, height=height, color='b', fill=False, linewidth=2)
        ax.add_artist(rectangleObj)
        annotate_string = 'Min: (' + str(min_x)+','+str(min_y)+')\n'+'Max: ('+str(max_x)+','+str(max_y) + ')'
        label = ax.annotate(annotate_string, xy=(center_x, center_y), fontsize=9, ha="center")

    plt.show()

draw_rectangles([list(itertools.chain(*ground_truth)), list(itertools.chain(*prediction))])