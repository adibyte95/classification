import sklearn as sk
import numpy as np
from PIL import Image
import pickle
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

no_of_images = 2000
input_KNN=[]
output_KNN=[]
'''
for i in range(0,12500):
	path = 'cats/cat.'+str(i)+'.jpg'
	img = cv2.imread(path)
	resized_image=cv2.resize(img,(200,200))
	print(np.array(resized_image,dtype="int32"))
	cv2.imshow('img',resized_image)
	cv2.waitKey(10)

'''
for i in range(0,no_of_images):
	path = 'cats/cat.'+str(i)+'.jpg'
	img = cv2.imread(path)
	resized_image=cv2.resize(img,(200,200))
	flat = np.array(resized_image,dtype="int32").flatten()
	input_KNN.append(flat)
	output_KNN.append(0)
	print('cat: ',i)


for i in range(0,no_of_images):
	path = 'dogs/dog.'+str(i)+'.jpg'
	img = cv2.imread(path)
	resized_image=cv2.resize(img,(200,200))
	flat = np.array(resized_image,dtype="int32").flatten()
	input_KNN.append(flat)
	output_KNN.append(1)
	print('dog: ',i)

X_train, X_test, y_train, y_test = train_test_split(input_KNN,output_KNN,test_size=0.33, random_state=6)
X_train = np.array(X_train)
X_test = np.array(X_test)

for i in range(1,50):
	clf = KNeighborsClassifier(n_neighbors=i) 
	model = clf.fit(X_train, y_train) 
	print('the predicted score is : ',clf.score(X_test,y_test))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
