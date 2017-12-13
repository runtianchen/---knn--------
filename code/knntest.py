import numpy as np
import operator

def file2matrix(filename):
	fr = open(filename)
	lines = fr.readlines()
	number_of_lines = len(lines)
	features = np.zeros((number_of_lines, 3))
	labels = []
	index = 0
	for line in lines:
		line = line.strip()
		data = line.split('\t')
		features[index, : ] = data[0:3]
		labels.append(int(data[-1]))
		index += 1
	return features,labels

def auto_norm(dataset):
    min_val=dataset.min(0)
    max_val=dataset.max(0)
    ranges = max_val-min_val
    m= dataset.shape[0]
    temp = np.tile(min_val,(m,1))
    norm_dataset=dataset-temp
    temp = np.tile(ranges,(m,1))
    norm_dataset = norm_dataset/temp

    return norm_dataset,ranges,min_val

def classify0(x,dataset,labels,k):
    dataset_size = dataset.shape[0]
    temp = np.tile(x,(dataset_size,1))
    diff_mat = temp - dataset
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_dist_indicies=distances.argsort()
    class_count ={}
    for i in range(k):
        voted_label=labels[sorted_dist_indicies[i]]
        class_count[voted_label]=class_count.get(voted_label,0)+1
        sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]

def dating_class_test():
    ratio = 0.1
    dataset,labels = file2matrix('dataSet.txt')
    norm_mat,ranges,min_vals=auto_norm(dataset)
    m=norm_mat.shape[0]
    num_test=int(m*ratio)
    err_count = 0.0
    for i in range(num_test):
        classifier_result=classify0(norm_mat[i,:],norm_mat[num_test:m,:],labels[num_test:m],3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifier_result,labels[i]))
        if classifier_result!=labels[i]:
            err_count+=1.0
    print("the total error rate is: %f"%(err_count/num_test))
    print(err_count)

if __name__ == '__main__':
    dating_class_test()