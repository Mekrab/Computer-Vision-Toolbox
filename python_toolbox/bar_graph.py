import matplotlib.pyplot as plt

#Def bar graph for output
def displayData(INPUT_1,INPUT_2):
    labels = ['VAR 1 X AXIS', 'VAR 2 X AXIS', 'VAR 3 X AXIS']
    x = np.arange(len(labels))
    bar_width = 0.40
    fig, ax = plt.subplots()
    bar_left = ax.bar(x - bar_width / 2, INPUT_1, bar_width, label='INPUT')
    bar_right = ax.bar(x + bar_width / 2, INPUT_2, bar_width, label='INPUT')
    ax.set_ylabel('YYYYYYYYY')
    ax.set_title('TITLE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def height_label(bars):                             # quick hight label for graph
        for bar_s in bars:
            height = bar_s.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar_s.get_x() + bar_s.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    height_label(bar_left)
    height_label(bar_right)
    fig.tight_layout()
    plt.show()

displayData(INPUT_1,INPUT_2)