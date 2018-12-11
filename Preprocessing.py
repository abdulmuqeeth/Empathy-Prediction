import csv
import numpy as np
from scipy import stats
from tempfile import TemporaryFile




def preprocess(data):

	smoking_index = data[0].index('Smoking')
	alcohol_index = data[0].index('Alcohol')
	punctuality_index = data[0].index('Punctuality')
	lying_index = data[0].index('Lying')
	internet_usage_index = data[0].index('Internet usage')
	gender_index = data[0].index('Gender')
	hand_index = data[0].index('Left - right handed')
	education_index = data[0].index('Education')
	child_index = data[0].index('Only child')
	village_index = data[0].index('Village - town')
	house_index = data[0].index('House - block of flats')
	empathy_index = data[0].index('Empathy')

	for i in range(1, len(data)):
		if(data[i][empathy_index]=='1' or data[i][empathy_index]=='2' or data[i][empathy_index]=='3'):
			data[i][empathy_index]='0'
		elif(data[i][empathy_index]=='4' or data[i][empathy_index]=='5'):
			data[i][empathy_index]='1'
		
		if(data[i][smoking_index]=='never smoked'):
			data[i][smoking_index] = 0
		elif(data[i][smoking_index]=='tried smoking'):
			data[i][smoking_index] = 1
		elif(data[i][smoking_index]=='former smoker'):
			data[i][smoking_index] = 2
		elif(data[i][smoking_index]=='current smoker'):
			data[i][smoking_index] = 3
		
		if(data[i][alcohol_index]=='never'):
			data[i][alcohol_index] =0
		elif(data[i][alcohol_index]=='social drinker'):
			data[i][alcohol_index] =1
		elif(data[i][alcohol_index]=='drink a lot'):
			data[i][alcohol_index]=2

		if(data[i][punctuality_index]=='i am often early'):
			data[i][punctuality_index] =0
		elif(data[i][punctuality_index]=='i am always on time'):
			data[i][punctuality_index] =1
		elif(data[i][punctuality_index]=='i am often running late'):
			data[i][punctuality_index]=2

		if(data[i][lying_index]=='never'):
			data[i][lying_index] =0
		elif(data[i][lying_index]=='only to avoid hurting someone'):
			data[i][lying_index] =1
		elif(data[i][lying_index]=='sometimes'):
			data[i][lying_index]=2
		elif(data[i][lying_index]=='everytime it suits me'):
			data[i][lying_index]=3

		if(data[i][internet_usage_index]=='no time at all'):
			data[i][internet_usage_index] =0
		elif(data[i][internet_usage_index]=='less than an hour a day'):
			data[i][internet_usage_index] =1
		elif(data[i][internet_usage_index]=='few hours a day'):
			data[i][internet_usage_index]=2
		elif(data[i][internet_usage_index]=='most of the day'):
			data[i][internet_usage_index] =0

		if(data[i][gender_index]=='male'):
			data[i][gender_index] =0
		elif(data[i][gender_index]=='female'):
			data[i][gender_index] =1

		if(data[i][hand_index]=='left handed'):
			data[i][hand_index] =0
		elif(data[i][hand_index]=='right handed'):
			data[i][hand_index] =1

		if(data[i][education_index]=='currently a primary school pupil'):
			data[i][education_index] =0
		elif(data[i][education_index]=='primary school'):
			data[i][education_index] =1
		elif(data[i][education_index]=='secondary school'):
			data[i][education_index] =2
		elif(data[i][education_index]=='college/bachelor degree'):
			data[i][education_index]=3
		elif(data[i][education_index]=='masters degree'):
			data[i][education_index] =4
		elif(data[i][education_index]=='doctorate degree'):
			data[i][education_index] =5


		if(data[i][child_index]=='no'):
			data[i][child_index] =0
		elif(data[i][child_index]=='yes'):
			data[i][child_index] =1

		if(data[i][village_index]=='city'):
			data[i][village_index] =0
		elif(data[i][village_index]=='village'):
			data[i][village_index] =1

		if(data[i][house_index]=='block of flats'):
			data[i][house_index] =0
		elif(data[i][house_index]=='house/bungalow'):
			data[i][house_index] =1


def most_freq_classifier(X,Y):
	most_freq = stats.mode(Y)[0][0]
	print(most_freq)
	if (most_freq ==0.0):
		predicted = np.zeros(len(Y))
	elif(most_freq==1.0):
		predicted = np.ones(len(Y))

	print('Accuracy :',np.mean(predicted==Y))

def normalize_data(data):

	data = data.T
	for i in range(len(data)):
		min = np.amin(data[i])
		max = np.amax(data[i])
		#print(min)
		#print(max)
		for j in range(len(data[0])):
			data[i][j] = (data[i][j]-min)/(max-min)
	
	data = data.T



def main():
	print('Preprocessing Module')

	print('parsing csv file')
	data=[]
	with open('responses.csv') as csv_file:
		csv_reader = csv.reader(csv_file, quotechar='"',delimiter=',',quoting=csv.QUOTE_ALL,skipinitialspace=True)
		for row in csv_reader:
			data.append(row)

	empathy_index = data[0].index('Empathy')
	print('Index of empathy column',empathy_index)

	print('Number of samples :',len(data))
	print('Number of attributes :',len(data[0]))
	
	print('Converting categorical data to numbers')
	preprocess(data)

	print('Finding mode of all columns')
	mode_array = stats.mode(data)[0][0]

	print('Filling empty cells with mode of respective columns')
	for i in range(len(data)):
		for j in range(len(data[0])):
			if(data[i][j]==''):
				data[i][j]=mode_array[j]

	#print(type(data))
	#print(len(data))

	data = np.array(data[1:]).astype(np.float)

	#Normalizing the data
	print('normalizing the data to scale between 0 and 1')
	normalize_data(data)
	print('data normalized')

	print('Shuffling the data')
	np.random.seed(4650)
	np.random.shuffle(data)
	#print(type(data))
	#print(len(data))
	#print(data)

	data2 = np.copy(data)
	data2 = data2.T

	print('Splitting Empathy column as Y and rest of data as X')
	Y = data2[empathy_index]
	X = np.delete(data,(empathy_index), axis=1)

	print('X :',X)
	print('Y :',Y)

	print('preprocessing complete')		

	#print('Using most frequent classifier')
	#most_freq_classifier(X,Y)		

	print('splitting data into train and test (80:20)')

	X_train = X[:int(len(X)*0.8)]
	Y_train = Y[:int(len(Y)*0.8)]

	#X_dev = X[int(len(X)*0.6):int(len(X)*0.8)]
	#Y_dev = Y[int(len(Y)*0.6):int(len(Y)*0.8)]

	X_test = X[int(len(X)*0.8):]
	Y_test = Y[int(len(Y)*0.8):]

	np.save('TrainX.npy', X_train)
	np.save('TrainY.npy', Y_train)
	#np.save('DevX.npy', X_dev)
	#np.save('DevY.npy', Y_dev)
	np.save('TestX.npy', X_test)
	np.save('TestY.npy', Y_test)

	print(len(X_train),len(X_test))
	print(len(Y_train),len(Y_test))

	

	# with open("output_data1.csv", "w") as out_file:
	# 	for i in range(len(data)):
	# 		for j in range(len(data[0])-1):
	# 			out_file.write(str(data[i][j])+ ",")
	# 		out_file.write(str(data[i][j+1])+ "\n")
			#out_file.write("\n")


if __name__  == "__main__":
	main()