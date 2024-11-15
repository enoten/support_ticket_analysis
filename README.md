<h2>Support Ticket Usecase</h2>

Here Support Ticket Usecase is considered.

The task is to classify the support tickets.

Support tickets are taken from the <a href="https://huggingface.co/datasets/phi-ai-info/support_tickets">support_ticket dataset</a>.
Dataset can be loaded either from HuggingFace repo or from the json files provided in this repo.

The code build to work with this dataset is doing following
<li> load data
<li> preprocess data
<li> create embedding for ticket descriptions
<li> perform dimensionality reduction 
<li> analyze ticket description with K-means clustering
<li> create a list of 2-grams (pairs of words) from the original descriptions
<li> create embedding collected pairs of words
<li> analyze pairs of words with K-means clustering
<li> create generic label for each ticket
<li> train and evaluate Logistic Regression classifier using Scikit Learn lib
<li> train and evaluate Deep Neural Network classifier using Keras lib

The end-to-end code is introduced in <a href="https://github.com/enoten/support_ticket_analysis/blob/main/support_tickets_problem.ipynb"> support_tickets_problem.ipynb </a>

Happy coding!
