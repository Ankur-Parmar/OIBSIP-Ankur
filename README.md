Hello, how are..?

This is my 1st Task which is given by OASIS INFOBYTE during my Internship Program..

Explanation:

First of all, I imported important libraries like pandas(used for data manipulation), seaborn(for Data Visualization), matplotlib(for creating static, animated, and interactive visualizations), scikit-Learn and some machine learning models with libraries etc.

And then, Step 1: Loaded a Iris Dataset using pd.read_csv()

Step 2: Exploration of Dataset using iris_data.head() - This will displays only first 5 rows of dataset to get overview of data.

Step 3: I also do Summary of Dataset using .descibe() - This will displays summary statistics of dataset (Mean, Standard Deviation, Min., Max., etc)

Step 4: Data Visualization - sns.pairplot(,hue='') - This will creates a pairplot means scatter plot matrix of the features colored by species to visualize relationships between features and species.
                                                   - And the "hue" specifies that which column to use for color coding.
                                                   - plt.show() - displays the plot.
Step 5 : Preprocess the Data using "iloc" - X: Contains all columns except the last one, which are the features used for training the model.
                                            y: Contains the last column, which is the target variable (species) to be predicted.
                                            
Step 6: Train the Model - LogisticRegression(max_iter=200): Initializes the Logistic Regression model. max_iter=200 sets the maximum number of iterations for convergence.
                        - model.fit(X_train, y_train): Trains the model using the training data.
                        
Step 7: Making Predictions - model.predict(X_test): Uses the trained model to make predictions on the test set.

Step 8: Evaluate the Model - classification_report(y_test, y_pred): Provides a detailed report on the model's performance, including precision, recall, and F1-score for each class.

And them to save the model use - joblib.dump(): Saves the trained model to a file named iris_classifier_model.pkl. This allows you to reuse the model without retraining.
To load the model - joblib.load(): Loads the model from the file iris_classifier_model.pkl.

And this one : model = LogisticRegression(max_iter=200)
               model.fit(X_train, y_train) --> THis will check if the model was trained by printing the model's coefficients.

And last - testing the loaded model with data.. I do prediction then the Prediction in the output is "0".. it means that he model has classified the given input as the class "Sentosa".
           In this if the prediction was 1 or 2 then the class will be "Versicolor" and "Virginica" respectively.

Thank You OASIS INFOBYTE..
I am very grateful to complete this task and I am very excited to do another task..
