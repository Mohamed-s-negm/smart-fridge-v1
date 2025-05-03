# smart-fridge-v1
This project is around dealing with sensors that can be used in fridge to add more functionalities.

I have added a file, generate_dataset.py to generate the dataset used to train the ml models. I have also added the dataset that is used.

I have added the ml_model.py which has the code for making ml models.

I have also added the generate_ml_models.py to generate the ml models needed in the project. I have also stored the generated ml models into a file.

NOTICE: There is a tag v1.0.1 which has an ml model which is made using RandomForest and so the should be downloaded and store along with the other ml models.

Now, the ml models are ready and now we head into the dl model that we need.

The food-101-dataset had to be reshaped so, I created the reshape-food-101.py file to do the job.

I have finished the dl image classification class model which is in dl_model.py. This file now can be used to create any image classification model.

First, I have trained the model using VGG16 and I didn't apply any fine-tuning and only for 10 epochs so I ended up with a 57% accuracy model which gave totally wrong information.
Then, I did just change the model into using ResNet50 and got even lower accuracy of 51%.

Now, I will use VGG16 along with full fine-tuning on the model and I will train it for 150 epochs.

I will also change the save function calling location from the end of the training to during the training to save the model which has the best accuracy at all time.
Second try, I got around 68% accuracy, the answers were different but still not right.

I changed a bit from using the full dataset into using half of it only 50 classes, I got accuracy of 75% and the answers for normal clear pictures were fine.

Now, I try to get rid of overfitting, I used stronger augementations for the train_transforms.
I made changes in the vgg16.classifier to ensure dropout is present. I added some weight_decay in the optimizer. I added label_smoothing in the criterion. I added gradient clipping in the train loop.
I enabled AMP in the training loop to improve the training speed, by adding scalar to the training method. lastly, I added confusion matrix to the test method.
I will train on the same dataset with only 50 classes.

The model didn't learn anything so I got rid of the label_smoothing and unfrooze more layers.
I also got rid of the added transformations which means back to normal augementations. I changed back some of the model initial definitions.
I got nearly 76 percent at the 9th epoch which is kinda fine but it showed overfitting.

This time I will add additional transformations.


