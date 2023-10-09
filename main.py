import csv
import os
import glob
import random
import math
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Get the current working directory
current_directory = os.getcwd()


# Now you can use the 'file_path' variable in your code



# Task 1: Splitting based on 'binary result' column in order to identify Benign and Attacks
# from our dataset file and then create new files based on the results

input_file = r"IanArffDataset.csv"
output_dir = os.path.join(current_directory, 'Datasets')
binary_result_column_name = 'binary result'
categorized_result_column_name = 'categorized result'

benign_output_file = os.path.join(output_dir, "Benign.txt")
attacks_output_file = os.path.join(output_dir, "Attacks.txt")

with open(input_file, 'r') as csv_file, \
     open(benign_output_file, 'w', newline='') as benign_file, \
     open(attacks_output_file, 'w', newline='') as attacks_file:

    reader = csv.reader(csv_file)
    benign_writer = csv.writer(benign_file)
    attacks_writer = csv.writer(attacks_file)

    header_row = next(reader)
    benign_writer.writerow(header_row)
    attacks_writer.writerow(header_row)

    binary_result_column_index = header_row.index(binary_result_column_name)

    for row in reader:
        for i, value in enumerate(row):
            if value == '?':
                row[i] = '-1'
        binary_result = row[binary_result_column_index]
        if binary_result == '0':
            benign_writer.writerow(row)
        elif binary_result == '1':
            attacks_writer.writerow(row)




# Task 2: Splitting 'Attacks.txt' based on 'categorized result' column(category of each attack)
# and creating new files based on the splitted data

attacks_input_file = attacks_output_file

with open(attacks_input_file, 'r') as attacks_file:
    reader = csv.reader(attacks_file)
    header_row = next(reader)

    categorized_result_column_index = header_row.index(categorized_result_column_name)

    categorized_output_files = [
        os.path.join(output_dir, f"Category{i}.txt") for i in range(1, 8)
    ]
    categorized_writers = [
        csv.writer(open(file, 'w', newline='')) for file in categorized_output_files
    ]

    for writer in categorized_writers:
        writer.writerow(header_row)  # Write the header row to each output file

    for row in reader:
        categorized_result = row[categorized_result_column_index]
        category_index = int(categorized_result) if categorized_result.isdigit() else 0

        if 0 < category_index <= 7:
            categorized_writer = categorized_writers[category_index - 1]
            categorized_writer.writerow(row)





# Task 3: Splitting data for each category and Benign into 80% and 20%
# for training our SVM model (80% data) and then use the rest of the data(20%) for evaluation

input_dir = os.path.join(current_directory, 'Datasets')
output_dir80 = os.path.join(current_directory, 'Datasets', 'Training_Data')
output_dir20 = os.path.join(current_directory, 'Datasets', 'Evaluation_Data')
file_suffixes = ['80', '20']

# Retrieve the files in the input directory
file_names = os.listdir(input_dir)

for file_name in file_names:
    if file_name != "Attacks.txt" and file_name.endswith(".txt"):
        input_file_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        input_data = []  # List to store the input data

        with open(input_file_path, 'r') as input_file:
            reader = csv.reader(input_file)
            header_row = next(reader)  # Read the header row from the input file

            for row in reader:
                input_data.append(row)  # Add each row to the input data list

        train_data, test_data = train_test_split(input_data, test_size=0.2, random_state=42)

        for suffix, data in zip(file_suffixes, [train_data, test_data]):
            if suffix == '80':
                output_dir = output_dir80
            else:
                output_dir = output_dir20

            output_file = os.path.join(output_dir, f"{base_name}_{suffix}.txt")

            with open(output_file, 'w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(header_row)  # Write the header row to the output file
                writer.writerows(data)  # Write the corresponding data to the output file


#Task 4: Sort the training and evaluation files based on the 'id' column

directory = os.path.join(current_directory, 'Datasets')

for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith("80.txt") or filename.endswith("20.txt"):
            file_path = os.path.join(root, filename)

            with open(file_path, 'r') as input_file:
                reader = csv.reader(input_file)
                header_row = next(reader)
                id_index = header_row.index('id')

                sorted_rows = sorted(reader, key=lambda row: int(row[id_index]))  # Sort rows based on the 'id' column

            with open(file_path + '.sorted', 'w', newline='') as output_file:
                writer = csv.writer(output_file)
                writer.writerow(header_row)  # Write header row

                for row in sorted_rows:
                    writer.writerow(row)  # Write sorted rows to output file

            # Replace original file with the sorted file
            os.replace(file_path + '.sorted', file_path)


#Task 5: Merge each category file created for training the SVM model with 
#the new Benign files that were created. 80% and 20% respectively and create new files
#and then sort the new files based on 'id'

directory = os.path.join(current_directory, 'Datasets', 'Training_Data')
output_directory = os.path.join(current_directory, 'Datasets', 'Training_Data', 'Merged_Files')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Find files that end in 80.txt (except Benign_80.txt)
for filename in os.listdir(directory):
    if filename.endswith("80.txt") and not filename.startswith("Benign"):
        file_path = os.path.join(directory, filename)

        # Merge the file with Benign_80.txt
        merged_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_merged.txt")
        benign_file_path = os.path.join(directory, "Benign_80.txt")

        with open(merged_file_path, 'w', newline='') as merged_file:
            writer = csv.writer(merged_file)

            # Write the header row from Benign_80.txt
            with open(benign_file_path, 'r') as benign_file:
                reader = csv.reader(benign_file)
                header_row = next(reader)
                writer.writerow(header_row)

            # Write the data from the current file and Benign_80.txt (except header row)
            with open(file_path, 'r') as current_file:
                reader = csv.reader(current_file)
                next(reader)  # Skip the header row

                for row in reader:
                    writer.writerow(row)

            # Append the data from Benign_80.txt (except header row)
            with open(benign_file_path, 'r') as benign_file:
                reader = csv.reader(benign_file)
                next(reader)  # Skip the header row

                for row in reader:
                    writer.writerow(row)


directory = os.path.join(current_directory, 'Datasets', 'Evaluation_Data')
output_directory = os.path.join(current_directory, 'Datasets', 'Evaluation_Data', 'Merged_Files')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Find files that end in 20.txt (except Benign_20.txt)
for filename in os.listdir(directory):
    if filename.endswith("20.txt") and not filename.startswith("Benign"):
        file_path = os.path.join(directory, filename)

        # Merge the file with Benign_20.txt
        merged_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_merged.txt")
        benign_file_path = os.path.join(directory, "Benign_20.txt")

        with open(merged_file_path, 'w', newline='') as merged_file:
            writer = csv.writer(merged_file)

            # Write the header row from Benign_20.txt
            with open(benign_file_path, 'r') as benign_file:
                reader = csv.reader(benign_file)
                header_row = next(reader)
                writer.writerow(header_row)

            # Write the data from the current file and Benign_20.txt (except header row)
            with open(file_path, 'r') as current_file:
                reader = csv.reader(current_file)
                next(reader)  # Skip the header row

                for row in reader:
                    writer.writerow(row)

            # Append the data from Benign_20.txt (except header row)
            with open(benign_file_path, 'r') as benign_file:
                reader = csv.reader(benign_file)
                next(reader)  # Skip the header row

                for row in reader:
                    writer.writerow(row)


output_dir = os.path.join(current_directory, 'Datasets')

# Iterate through all files and subdirectories within output_dir
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('merged.txt'):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, f"{os.path.splitext(file)[0]}_final.txt")

            # Read the input file into a DataFrame with low_memory=False
            df = pd.read_csv(input_file, low_memory=False)

            # Convert 'id' column to numeric type
            df['id'] = pd.to_numeric(df['id'], errors='coerce')

            # Sort the DataFrame based on the 'id' column
            df = df.sort_values('id')

            # Remove unnecessary columns
            df = df.drop(['id', 'address', 'time'], axis=1)

            # Write the modified DataFrame to the output file
            df.to_csv(output_file, index=False)




#Kernel Selection

#SVM model - Category 1 - Sigmoid

#Read the training data file:
train_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Training_Data', 'Merged_Files', 'Category1_80_merged_final.txt'))

#Extract the input features (x_train) and the output (y_train):
x_train = train_data.iloc[:, :-3]
y_train = train_data['binary result']

# Store the feature names separately
feature_names = x_train.columns.tolist()

# Create a scaler object
scaler = StandardScaler()

# Scale the input features
x_train_scaled = scaler.fit_transform(x_train)

#Create an instance of the SVM classifier:
svm_model = SVC(kernel='sigmoid', gamma=0.01, coef0=0.0, C=1.0)

#Fit the SVM model to the training data:
svm_model.fit(x_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(svm_model, os.path.join(current_directory, 'Datasets', 'SVM_Models', 'Sigmoid_Kernel', 'svm_model_category1_sigmoid.pkl'))



#Attack Category 1 - Sigmoid Kernel

#Load the saved model from the file
loaded_svm_model = joblib.load(os.path.join(current_directory, 'Final_Data', 'SVM_Models', 'Sigmoid_Kernel', 'svm_model_category1_sigmoid.pkl'))

#Read the test data file:
test_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Evaluation_Data', 'Merged_Files', 'Category1_20_merged_final.txt'))

#Extract the input features for testing (x_test):
x_test = test_data.iloc[:, :-3]
y_test = test_data['binary result']

#Predict the output labels using the trained SVM model:
y_pred = loaded_svm_model.predict(x_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Sigmoid Kernel")
plt.show()

# Create a DataFrame with custom labels
cm_df = pd.DataFrame([[f"True Negative: {cm[0, 0]}", f"False Positive: {cm[0, 1]}"],
                    [f"False Negative:  {cm[1, 0]}", f"True Positive: {cm[1, 1]}"]], index=['', ''], columns=['', ''])

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_df)

#Calculate the accuracy of the SVM model:
accuracy = accuracy_score(test_data['binary result'], y_pred)
print("\n\nAccuracy:", accuracy)

# Calculate classification metrics
report = classification_report(test_data['binary result'], y_pred, zero_division=1)

# Print the classification report
print("\n\nClassification Report:")
print(report)

# Extract values from confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate True Positive Rate and True Negative Rate
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

# Calculate Balanced Accuracy
balanced_accuracy = (TPR + TNR) / 2
print("Balance Accuracy: ", balanced_accuracy)


# Same procedure for the other kernels as well(adjust accordingly)



#Task 6: Train the SVM models
#Task 7: Test the accuracy of the SVM model using the evaluation data
#and generate the confusion matrix,classification metrics 
#and then calculate the balance accuracy


#SVM model - Category 1

#Read the training data file:
train_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Training_Data', 'Merged_Files', 'Category1_80_merged_final.txt'))

#Extract the input features (x_train) and the output (y_train):
x_train = train_data.iloc[:, :-3]
y_train = train_data['binary result']

# Store the feature names separately
feature_names = x_train.columns.tolist()

# Create a scaler object
scaler = StandardScaler()

# Scale the input features
x_train_scaled = scaler.fit_transform(x_train)

#Create an instance of the SVM classifier:
svm_model = SVC(kernel='linear', C=1.0)

#Fit the SVM model to the training data:
svm_model.fit(x_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(svm_model, os.path.join(current_directory, 'SVM_Models', 'Normal', 'svm_model_category1_scaled.pkl'))


#Attack Category 1

#Load the saved model from the file
loaded_svm_model = joblib.load(os.path.join(current_directory, 'SVM_Models', 'Normal', 'svm_model_category1_scaled.pkl'))

#Read the test data file:
test_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Evaluation_Data', 'Merged_Files', 'Category1_20_merged_final.txt'))

#Extract the input features for testing (x_test):
x_test = test_data.iloc[:, :-3]
y_test = test_data['binary result']

#Predict the output labels using the trained SVM model:
y_pred = loaded_svm_model.predict(x_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Category 1 (NMRI)")
plt.show()

# Create a DataFrame with custom labels
cm_df = pd.DataFrame([[f"True Negative: {cm[0, 0]}", f"False Positive: {cm[0, 1]}"],
                    [f"False Negative:  {cm[1, 0]}", f"True Positive: {cm[1, 1]}"]], index=['', ''], columns=['', ''])

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_df)

#Calculate the accuracy of the SVM model:
accuracy = accuracy_score(test_data['binary result'], y_pred)
print("\n\nAccuracy:", accuracy)

# Calculate classification metrics
report = classification_report(test_data['binary result'], y_pred, zero_division=1)

# Print the classification report
print("\n\nClassification Report:")
print(report)

# Extract values from confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate True Positive Rate and True Negative Rate
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

# Calculate Balanced Accuracy
balanced_accuracy = (TPR + TNR) / 2
print("Balance Accuracy: ", balanced_accuracy)



# Same procedure for the rest of the SVM models(different input/output paths and names)


#Data Balancing

#Under Sampling

# Set the directory path
directory = os.path.join(current_directory, 'Datasets', 'Training_Data')

# Create a new directory for training datasets
training_directory = os.path.join(directory, 'Undersampling')
if not os.path.exists(training_directory):
    os.makedirs(training_directory)

# Get all files ending with "80.txt" in the directory
files_80 = [file for file in os.listdir(directory) if file.endswith('80.txt') and file != 'Benign_80.txt']

for file_name in files_80:
    # Read the current file and get the number of rows
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        file_data = file.readlines()
    number_of_rows = len(file_data)
    
    # Extract random rows from Benign_80.txt
    benign_file_path = os.path.join(directory, 'Benign_80.txt')
    with open(benign_file_path, 'r') as benign_file:
        benign_data = benign_file.readlines()
    random_rows = random.sample(benign_data, int(number_of_rows * 0.2))
    
    # Create a new file in the training directory and save the extracted rows
    output_file_name = file_name.replace('.txt', '_merged.txt')
    output_file_path = os.path.join(training_directory, output_file_name)
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(file_data)
        output_file.writelines(random_rows)


directory = os.path.join(current_directory, 'Datasets', 'Evaluation_Data')

# Create a new directory for training datasets
training_directory = os.path.join(directory, 'Undersampling')
if not os.path.exists(training_directory):
    os.makedirs(training_directory)

# Get all files ending with "20.txt" in the directory
files_20 = [file for file in os.listdir(directory) if file.endswith('20.txt') and file != 'Benign_20.txt']

for file_name in files_20:
    # Read the current file and get the number of rows
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        file_data = file.readlines()
    number_of_rows = len(file_data)
    
    # Extract random rows from Benign_20.txt
    benign_file_path = os.path.join(directory, 'Benign_20.txt')
    with open(benign_file_path, 'r') as benign_file:
        benign_data = benign_file.readlines()
    random_rows = random.sample(benign_data, int(number_of_rows * 0.2))
    
    # Create a new file in the training directory and save the extracted rows
    output_file_name = file_name.replace('.txt', '_merged.txt')
    output_file_path = os.path.join(training_directory, output_file_name)
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(file_data)
        output_file.write('\n')  # Add a new line
        output_file.writelines(random_rows)


output_dir = os.path.join(current_directory, 'Datasets', 'Training_Data', 'Undersampling')

# Iterate through all files and subdirectories within output_dir
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('merged.txt'):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, f"{os.path.splitext(file)[0]}_final.txt")

            # Read the input file into a DataFrame with low_memory=False
            df = pd.read_csv(input_file, low_memory=False)
        
            # Convert 'id' column to numeric type
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
            # Sort the DataFrame based on the 'id' column
            df = df.sort_values('id')

            # Remove unnecessary columns
            df = df.drop(['id', 'address', 'time'], axis=1)

            # Write the modified DataFrame to the output file
            df.to_csv(output_file, index=False)


output_dir = os.path.join(current_directory, 'Datasets', 'Evaluation_Data', 'Undersampling')

# Iterate through all files and subdirectories within output_dir
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith('merged.txt'):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, f"{os.path.splitext(file)[0]}_final.txt")

            # Read the input file into a DataFrame with low_memory=False
            df = pd.read_csv(input_file, low_memory=False)
        
            # Convert 'id' column to numeric type
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
            # Sort the DataFrame based on the 'id' column
            df = df.sort_values('id')

            # Remove unnecessary columns
            df = df.drop(['id', 'address', 'time'], axis=1)

            # Write the modified DataFrame to the output file
            df.to_csv(output_file, index=False)


#SVM model - Category 1 - Undersampling

#Read the training data file:
train_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Training_Data', 'Undersampling', 'Category1_80_merged_final.txt'))

scaler = StandardScaler()

#Extract the input features (x_train) and the output (y_train):
x_train = train_data.iloc[:, :-3]
y_train = train_data['binary result']

x_train_scaled = scaler.fit_transform(x_train)

#Create an instance of the SVM classifier:
svm_model = SVC(kernel='linear', C=1.0)

# Continue with model fitting
svm_model.fit(x_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(svm_model, os.path.join(current_directory, 'SVM_Models', 'Undersampling', 'svm_model_category1_undersampling.pkl'))


#Attack Category 1 - Undersampling

#Load the saved model from the file
loaded_svm_model = joblib.load(os.path.join(current_directory, 'SVM_Models', 'Undersampling', 'svm_model_category1_undersampling.pkl'))

#Read the test data file:
test_data = pd.read_csv(os.path.join(current_directory, 'Datasets', 'Evaluation_Data', 'Undersampling', 'Category1_20_merged_final.txt'))

#Extract the input features for testing (x_test):
x_test = test_data.iloc[:, :-3]
y_test = test_data['binary result']

#Predict the output labels using the trained SVM model:
y_pred = loaded_svm_model.predict(x_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Undersampling")
plt.show()

# Create a DataFrame with custom labels
cm_df = pd.DataFrame([[f"True Negative: {cm[0, 0]}", f"False Positive: {cm[0, 1]}"],
                    [f"False Negative:  {cm[1, 0]}", f"True Positive: {cm[1, 1]}"]], index=['', ''], columns=['', ''])

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_df)

#Calculate the accuracy of the SVM model:
accuracy = accuracy_score(test_data['binary result'], y_pred)
print("\n\nAccuracy:", accuracy)

# Calculate classification metrics
report = classification_report(test_data['binary result'], y_pred, zero_division=1)

# Print the classification report
print("\n\nClassification Report:")
print(report)

# Extract values from confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Calculate True Positive Rate and True Negative Rate
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

# Calculate Balanced Accuracy
balanced_accuracy = (TPR + TNR) / 2
print("Balance Accuracy: ", balanced_accuracy)


# Same procedure for the other balancing techniques(adjust accordingly)