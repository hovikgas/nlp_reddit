# Classification of the /r/ADHD and /r/Anxiety SubReddits

## Executive Summary

This analysis uses data from 2 SubReddits: /r/ADHD and /r/Anxiety to train a Natural Language Processing (NLP) classifier on which subreddit a given post comes from. The purpose was to see if the language used in the two subreddits is different, and if so, how, since ADHD and Generalized Anxiety Disorder have many overlapping symptoms and ADHD is often misdiagnosed as a result. I start off by defining my problem statement. I then provide some background information, followed by the results of my analysis.

## Problem Statement

Attention deficit hyperactivity disorder (ADHD) is one of the most frequently diagnosed disorders in children, yet there is much disagreement around its correct diagnosis [(Ford-Jones, 2015).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4443828/) According to the Anxiety and Depression Association of America (ADAA), about **half** of adults with ADHD also suffer from an anxiety disorder [(ADAA, 2019).](https://adaa.org/understanding-anxiety/related-illnesses/other-related-conditions/adult-adhd) Overlapping symptoms of comorbid psychiatric conditions often complicate an already difficult diagnosis. Fortunately, there are online communities where users gather to share their experiences, ask questions, and support one another. Social media data has been shown to be an untapped resource for knowledge on mental illnesses. For example, Twitter has been used to create classifiers that recognize depression in users [(De Choudhury et al., 2013).](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icwsm_13.pdf) Two such untapped resources exist on Reddit, in the /r/ADHD and /r/Anxiety subreddits. I believe that this analysis may be useful in assisting stakeholders to compare and contrast ADHD and Anxiety, identifying various overlaps and differences between the two groups. Some interested stakeholders might be healthcare professionals, diagnosed individuals, or even undiagnosed individuals who have loved ones suffering from either ADHD, anxiety, or both. 

## Background 

Reddit is a social media network where registered users can create posts to a broader community. Posts are hosted in topic-specific forums, called subreddits, which are created by users for anyone interested in that topic. There are over 1.2 million subreddits, ranging from sports, to politics, and everything in between. Users can join any number of subreddits and participate in discussions. The ADHD subreddit has nearly 400,000 subscribers and the official description reads, "A place where people with ADHD and their loved ones can interact with each other exchanging stories, struggles, and strategies. Weekly threads to plan and notice the positive in our lives. Many of the 300k+ users express they 'feel at home' and 'finally found a place where people understand them'." The Anxiety subreddit has 245,000 subscribers and the official description reads, "Discussion and support for sufferers and loved ones of any anxiety disorder." 

## Analysis

To download the posts from the two subreddits, I used the PushShift API. I then cleaned the data by removing URLs and non-letter characters (e.g. slashes, line breaks, etc.). I then engaged in some Exploratory Data Analysis (EDA). One of the highlights from the EDA include the insight that posts from the ADHD subreddit tend to be shorter than those from the Anxiety subreddit. The average ADHD post was 165 words, compared to 185 for Anxiety. The picture below shows the distribution of the length of posts from the two subreddits. 

![wordcount](./assets/wordcount.png "Word Count") 

As we can see, both distributions are skewed to the right, with the vast majority of posts being under 1000 words, and only a handful having more than that. 

---

#### Below are some of the most commonly used and interesting words in the two subreddits (after cleaning):

| ADHD | Anxiety |
| ---- | ------- |
| Time: 61% | Feel: 91% |
| Feel: 55% | Know: 70% |
| Know: 53% | Time: 64% |
| Work: 44% | Want: 48% |
| Things: 42% | People: 46% |
| Medication: 25% | Medication: 12% |


It's no surprise that the most frequently used word in the ADHD subreddit is 'time,' since time management is one of the biggest struggles for those with ADHD. Interestingly, time is the third most commonly used word in the anxiety subreddit, which highlights the issue of overlap mentioned in the problem statement. In fact, looking at the top 20 terms in each subreddit, we see that 14 of them overlap.

  * time
  * feel
  * know
  * work
  * things
  * day
  * want
  * life
  * people
  * going
  * got
  * think
  * need
  * help

Next, I moved onto some more advanced natural language processing (NLP). I add some custom stop words that aren't included in the default SciKit-Learn library (stop words are very common words that do not convey much meaning, which we explicitly tell our models to ignore). I used both CountVectorizer and TF-IDF (term frequency-inverse document frequency). The difference between the two is that, instead of simply counting the frequencies of our words, TF-IDF tries to distinguish their importance by assigning weights proportional to the number of times a word appear in a specific post, offset by the number of posts that use that word. For both, though, I set a minimum document frequency of 5 (i.e. words must be in at least 5 posts to be included). I also set the max_df to 1.0, meaning that no words will be left out because they are too common. However, the TF-IDF algorithm weights very common words accordingly.

I also conducted some latent semantic analysis (LSA) using singular value decomposition (SVD). LSA using SVD is similar to principal components analysis (PCA), but does not center the data. In Python, computing the SVD is similar to any other sklearn preprocessing method, where we fit and transform our data. Normally, we would want to split our data into training and testing sets before running our decomposition. However, I'm not worried too about the variance of our validation data impacting our components. As we'll see later, the results do not differ much between models. 

![first two components](./assets/components_12.png)

Component one seems to be regarding the time management issue that I previously touched upon. People with ADHD knowing that there is work to do, and really want to do it, but struggle with managing their time and feel bad for not being able to handle the little, everyday things that other people take for granted. Component 2 seems to be regarding various medication and their prescription/dosage. Interestingly, we can see negative loadings for people, friends, know, think, want, talk, and anxious, because they don't relate to the medication. Above, I plotted these two components to see if any discernible separation exists, but there doesn't seem to be very good separation. Below is the same plot for the second and third components. 

![second and third components](./assets/components_23.png)

For the second and third compondents there seems to be a bit more separation, but there is still a lot of overlap, which is, again, not surprising. Moving on, I tested out logistic regression and random forests in my modeling. If you're unfamiliar with Random Forests, the name comes from its relation to Decision Trees, which is essentially a classifier based on splitting up the data using many if-statements. Random Forests classification uses many (i.e. bagged) decision trees, bootstraps them (i.e. sampling with replacement), and modifies the tree learning algorithm so that at each split in the learning process, a random subset of features (in this case words) is used.

Surprisingly, logistic regression performed well. On the training set, the accuracy rate was 89% and on the test set it was 84%. The Random Forests Classifier overfit on the training set, getting an accuracy of over 99%, but did slightly worse on the test set than the logistic regression model, getting only 81%. The logistic regression model's specificity (i.e. true negative rate) is 81%, meaning that of the posts that were actually from the anxiety subreddit, 81% were correctly classified. The logistic regression model's sensitivity (i.e. true positive rate) is 89%, meaning that of the posts that were actually from the ADHD subreddit, 89% were correctly classified. To better visualize this tradeoff between sensitivity and specificity, I will graph the receiver operating characteristic (ROC) curves for each model below.

![important components](./assets/impcomp.png)

Interestingly, component 2, which I previously described as the medication-related component provided the most information gain. It was followed by the third component, which seemed to be regarding job-related panic attacks and general anxiety towards work. Regardless, because these components are not as intuitive to look at as the words themselves, so I will rerun the models without using singular value decomposition to provide this same graph for the most important words.

![top 20 features](./assets/top20feats.png)

Here we can see that panic, anxious, and attack are the top 3 important features in our random forests model. This is interesting, because these were not the words we focused on during the EDA process. However, the rest of the words are those we expect, such as those related to medication and emotions.

#### AUC-ROC Curve for Logistic Regression Model

![AUCROCLogReg](./assets/AUC_ROC_LogReg.png)

#### AUC-ROC Curve for Random Forests Model

![AUCROCLogReg](./assets/AUC_ROC_RF.png)

As we can see, both of the AUC-ROC curves are quite similar, although the one for the logistic regression model looks a bit smoother. Regardless, it's ROC-AUC score was .93, compared with .90 for the random forests model. As such, if I had to pick one, I would choose the logistic regression model for my final production model. 

## Conclusion

This preliminary and exploratory analysis used data from 2 SubReddits: /r/ADHD and /r/Anxiety to train two Natural Language Processing (NLP) classifiers on which subreddit a given post came from: Logistic Regression and Random Forests. Although both models performed well, logistic regression performed slightly better. Given it's vastly faster compute time, and the fact that it is the more parsimonous of the two, I would choose the logistic regression model for my final production model. 

It's a fine balancing act deciding which words to remove and which to keep. In the future I would like to test out the removal of more words, as well as explore bi-grams and tri-grams (e.g. panic attack, generalized anxiety disorder, social anxiety, health anxiety, etc.). Also, I would like to test other models as well, such as gradient boosting and K-Nearest Neighbors. Ultimately, this is not a causal model, nor am I a medical expert. My aim has simply been to start to shed light on, and extend to outsiders, the dialogues that these hundreds of thousands of Reddit users have been engaging in with one another. If even one person feels less alone, or one medical professional more informed, I would consider this project a success. 