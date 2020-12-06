
# Function for generating histogram in python
def generate_histogram(image_dict, number_bins=8):
    histogram_dict = dict()
    for filename in image_dict:
        image = image_dict[filename]
        hist0 = cv2.calcHist([image], [0], None, [number_bins], [0, 256])
        hist1 = cv2.calcHist([image], [1], None, [number_bins], [0, 256])
        hist2 = cv2.calcHist([image], [2], None, [number_bins], [0, 256])
        overall_hist = np.concatenate([hist0, hist1, hist2]).ravel()

        hist = overall_hist / overall_hist.sum()
        histogram_dict[filename] = hist
    return histogram_dict

# Function for comparing histograms
def compare_histogram(image_name, hist_base, histogram_dict):
    results_dict = dict()
    for (methodName, method, reverse) in OPENCV_METHODS:
        results = {k: cv2.compareHist(hist_base, hist, method) for (k, hist) in sorted(histogram_dict.items())}
        # print(type(results))
        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        # print(type(results))

        results_dict[methodName] = results

    return results_dict

# For loop to print out scores
for (methodName, method, reverse) in OPENCV_METHODS:
    plt.figure(figsize=(10, 6))

    for index, class_name in enumerate(class_list):
        comparison_list = [x for (x,y) in target_class_compare_dict[class_name][class_name][methodName]]
        plt.plot(comparison_list, marker='',linewidth=1, alpha=0.4) #label = x_axis_values)
    plt.legend(class_list, loc='upper right')
    plt.ylabel('score')
    plt.xlabel('image count')
    plt.title('Compare ' + methodName)
    plt.show()

# Function for calculate score range for histogram
def calculate_score_range(histogram_scores_tuple):
    scores = [score for (score,filename) in histogram_scores_tuple]
    avg_score = sum(scores)/len(scores)
    median_score = scores[int(len(scores)/2)]   # Assume the scores are sorted
    max_score = max(scores)
    min_score = min(scores)
    return (avg_score, median_score, max_score, min_score)