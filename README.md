Download Link: https://assignmentchef.com/product/solved-b555-programming-project1-sentiment-analysis-through-naive-bayes
<br>
In this assignment you will implement the Naive Bayes algorithm with maximum likelihood and MAP solutions and evaluate it using cross validation on the task of sentiment analysis (as in identifying positive/negative product reviews).

2           Text Data for Sentiment Analysis

We will be using the “Sentiment Labelled Sentences Data Set”<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> that includes sentences labelled with sentiment (1 for positive and 0 for negative) extracted from three domains imdb.com, amazon.com, yelp.com. These form 3 datasets for the assignment.

Each dataset is given in a single file, where each example is in one line of that file. Each such example is given as a list of space separated words, followed by a tab character (t), followed by the label, and then by a newline (
). Here is an example from the yelp dataset:

Crust is not good.                          0

The data, which is hosted by the UCI machine learning repository, is linked through the course web page.

<h1>3           Implementation</h1>

<h2>3.1         Naive Bayes for Text Categorization</h2>

In this assignment you will implement “Naive Bayes for text categorization” as discussed in class. In our application every “document” is one sentence as explained above. The description in this section assumes that a dataset has been split into separate train and test sets.

Given a training set for Naive Bayes you need to parse each example and record the counts for class and for word given class for all the necessary combinations. These counts constitute the learning process since they determine the prediction of Naive Bayes (for both maximum likelihood and MAP solutions).

Now, given the test set, you parse each example, calculate the scores for each class and test the prediction. Note that products of small numbers (probabilities) will quickly lead to underflow problems. Due to that you should work with sum of log probabilities instead of product of probabilities. Recall that <em>a </em>· <em>b </em>· <em>c &gt; d </em>· <em>e </em>· <em>f </em>iff log<em>a </em>+ log<em>b </em>+ log<em>c &gt; </em>log<em>d </em>+ log<em>e </em>+ log<em>f </em>so that working with the logarithms is sufficient.

<strong>Important point for prediction: </strong>If a word in a test example did not appear in the training set at all (i.e. in any of the classes), then simply skip that word when calculating the score for this example. However, if the word did appear with some class but not the other then use the counts you have (zero for one class but non zero for the other).

<h2>3.2         Maximum Likelihood and MAP Solutions</h2>

Recall that we are using the feature of type “token in document is word <em>w</em>”, so that each “token feature” has as many values as words in the vocabulary (all words in training files). The maximum likelihood (and MAP) estimates of parameters are given by the solution for a Discrete distribution (with a Dirichlet prior) for its parameters.

The maximum likelihood estimate of <em>p</em>(<em>w</em>|<em>c</em>) for word <em>w </em>and class <em>c </em>is <sup>#(</sup><sub>#(</sub><em><sup>w</sup><sub>c</sub></em><sup>∧</sup><sub>)</sub><em><sup>c</sup></em><sup>) </sup>where #(<em>w </em>∧<em>c</em>) is the number of word tokens in examples of class <em>c </em>that are the word <em>w </em>and #(<em>c</em>) is the number of word tokens in examples of class <em>c</em>.

If we use a prior with parameter vector where all entries are equal to <em>m </em>+ 1, that is, (<em>m </em>+ 1)<strong>1</strong>, the effect is that of adding a pseudo count of <em>m </em>to all entries. In this case, the MAP estimate of <em>p</em>(<em>w</em>|<em>c</em>) is <sup>#(</sup><sub>#(</sub><em><sup>w</sup><sub>c</sub></em><sub>)+</sub><sup>∧<em>c</em>)+</sup><em><sub>mV</sub><sup>m </sup></em>where <em>V </em>is the vocabulary size and other parameters are as above. For example, if <em>m </em>= 1 and <em>V </em>= 1000, and out of 10000 word locations in examples of class <em>c </em>the word <em>w </em>appeared 100 times, the probability is estimated to be .

This estimate is often referred to as “smoothing” in the literature because it smoothes out the maximum likelihood values and avoids 0/1 extreme solutions. Note that the maximum likelihood solution is simply the special case of smoothing with m=0.

<h2>3.3         Cross Validation</h2>

Implement code to read and parse a dataset and prepare it for 10-fold stratified cross validation.

The pseudo-code for generating such folds is given in the lecture slides, and briefly explained here.

The simplest way to get random folds is to randomly permute the order of examples and then split the permuted indices sequentially. To get stratified folds you need to do this for each class separately and then put together the different portions. For example, assume the initial classes are 1<em>,…,</em>10 (positive examples) and 11<em>,…,</em>20 (negative examples) and that our permutations are [3<em>,</em>2<em>,</em>4<em>,</em>5<em>,</em>7<em>,</em>8<em>,</em>1<em>,</em>9<em>,</em>6<em>,</em>10] and [17<em>,</em>13<em>,</em>14<em>,</em>18<em>,</em>11<em>,</em>15<em>,</em>12<em>,</em>16<em>,</em>19<em>,</em>20]. Then for 2-fold cross validation we produce the folds [3<em>,</em>2<em>,</em>4<em>,</em>5<em>,</em>7<em>,</em>17<em>,</em>13<em>,</em>14<em>,</em>18<em>,</em>11] and [8<em>,</em>1<em>,</em>9<em>,</em>6<em>,</em>10<em>,</em>15<em>,</em>12<em>,</em>16<em>,</em>19<em>,</em>20]. In <em>k</em>-fold cross validation we generate <em>k </em>train/test splits where the <em>i</em>th split is given by training on all but the <em>i</em>th portion and testing on the <em>i</em>th portion.

It is important to randomize the ordering in the final training sets in case the algorithm is sensitive to example ordering. While Naive Bayes is not sensitive to ordering, our learning curve experiment of the next section is sensitive. For example, if we use initial portions of the fold [3<em>,</em>2<em>,</em>4<em>,</em>5<em>,</em>7<em>,</em>17<em>,</em>13<em>,</em>14<em>,</em>18<em>,</em>11] in that order it will only include positive examples, which will skew the results.

<h2>3.4         Learning Curves with Cross Validation</h2>

Learning curves evaluate how the predictions improve with increasing train set size. To measure this with cross validation we follows the following procedure. First generate the folds for cross validation, call these train<em>i</em>, test<em>i</em>, for <em>i </em>= 1<em>,…,k</em>. Say train <em>i </em>has <em>N </em>examples. Then use subsamples of train<em>i </em>(you can use an initial portion if the data was randomized) of sizes 0<em>.</em>1<em>N,</em>0<em>.</em>2<em>N,…,</em>0<em>.</em>9<em>N,N </em>as train sets and evaluate the prediction on test<em>i </em>measuring the accuracy in each case. Repeat this for each <em>i </em>and then calculate the average and standard deviation for each size. This constitutes the learning curve.

<h1>4           Experiments</h1>

Once all the above is implemented please run the following tests and report the results. You are requested to run Naive Bayes many times on combinations of datasets, parameters, and folds. With a reasonable implementation the overall run time should not be high (≤ 1min). But please plan ahead to make sure you can complete the assignment on time.

<ul>

 <li>For each of the 3 datasets run stratified cross validation to generate learning curves for Naive Bayes with <em>m </em>= 0 and with <em>m </em>= 1. For each dataset, plot averages of the accuracy and standard deviations (as error bars) as a function of train set size. It is insightful to put both <em>m </em>= 0 and <em>m </em>= 1 together in the same plot. What observations can you make about the results?</li>

 <li>Run stratified cross validation for Naive Bayes with smoothing parameter <em>m </em>= 0<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>2<em>,…,</em>0<em>.</em>9 and 1<em>,</em>2<em>,</em>3<em>, …,</em>10 (i.e., 20 values overall). Plot the cross validation accuracy and standard deviations as a function of the smoothing parameter. What observations can you make about the results?</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences