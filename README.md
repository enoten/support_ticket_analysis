<h2>Support Ticket Usecase</h2>

Here Support Ticket Usecase is considered.

The task is to classify the support tickets.

<h2>Dataset</h2>

Support tickets are taken from the <a href="https://huggingface.co/datasets/phi-ai-info/support_tickets">support_ticket dataset</a>.

Dataset can be loaded either from HuggingFace repo or from the json files provided in this repo.<br>

This dataset contains 60 samples. Each tickets resembles a support ticket. <br>
Two main fields are subject and description. Subject is a short title for the ticket, <br>
while description contains extended description of the issue.<br>

The thrid field is a key_phase, which indicate key phrase of the ticket.<br>

You can all split the entire dataset into thress classes:<br>
<li> access
<li> user
<li> disk
<br>
Key phrases for each class are as follows:<br>
<b>{<br>
'access': ['grant access', 'revoke access', 'access profile'],<br>
'user': ['add user', 'delete user', 'modify user','create user'],<br>
'disk': ['disk space', 'disk error', 'disk full']<br>
}</b>

<h2>Use case code</h2>
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

The end-to-end code is introduced in <a href="https://github.com/enoten/support_ticket_analysis/blob/main/support_tickets_problem.ipynb"> support_tickets_problem.ipynb </a><br>

Happy coding!
