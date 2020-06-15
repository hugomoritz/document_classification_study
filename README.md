# Document Classification Study

The program is a development of testing five different machine learning algorithms for document classification with Python in the scikit-learn library.

To run the program: 

1. Make sure to have the necessary libraries installed mentioned in license.txt.
2. Compile main.py
   If errors occur because of the stopwords file path. Make sure to change the syntax to work for the specficic operating system. 
   The current solution is valid for Windows.
3. Run the program with main-function. 
   Notice that the input parameter must be a file of type .csv . 
   The data in the file should be at least two columns namned 'Content' and 'Labels'. If not, 
   make sure the text data column is the first and class label column is the second, and change 
   the the code in doc_rep.py in the tfidf function in row 19 and 20.
