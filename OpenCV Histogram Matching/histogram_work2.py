
# Function to  plotting histograms RBG
def plotting_hist(images, query):
    plt.imshow(cv2.cvtColor(images[query], cv2.COLOR_BGR2RGB))
    plt.title(query)
    plt.show()
    color = ('r', 'g', 'b')

    for i, col in enumerate(color):
        hist = cv2.calcHist([images[query]], [i], None, [300], [40, 230])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
        plt.ylim([0, 10000])
        plt.title(query)
    plt.show()
    return

# Function for searching "Correlation", "Intersection" in python
def search(classes, query):
    method_results = {}
    for position, (methodName, method) in enumerate(OPENCV_METHODS):
        # initialize the results dictionary and the sort
        # direction
        results = {}
        reverse = False

        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        # loop over the index

        for (k, hist) in classes.items():
            # compute the distance between the two histograms
            # using the method and update the results dictionary
            d = cv2.compareHist(classes[query], hist, method)
            results[k] = d

            # print(results)
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        method_results[methodName] = results

    return method_results

# For Loop to look into Query vs. Target
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mystery_vs_target_dic["mystery1.png"])
plt.axis("off")
# initialize the results figure
fig = plt.figure(figsize=(200, 200), dpi=150)
fig = plt.figure("Results:")
fig.suptitle(m_name[0], fontsize=20)
for (i, (v, k)) in enumerate(mystery_score["Correlation"]):
    # show the result
    ax = fig.add_subplot(1, len(mystery_vs_target_dic), i + 1)

    ax.set_title("%.2f" % (v))
    plt.imshow(mystery_vs_target_dic[k])
    plt.tight_layout(pad=400)
    plt.axis("off")
    plt.tight_layout()
# show the OpenCV methods
plt.show()