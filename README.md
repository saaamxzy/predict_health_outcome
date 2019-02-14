# predict_health_outcome
Models used to predict American Health outcomes written in Matlab.

## Feature files
training_data.csv

Features and labels are contained in the file training_data.csv. Here is the description of each column in the file.

FIPS Code: Column 1 containts the unique FIPS code of each county. Each row is the data of one county, and there are 1019 counties.

Features: Columns 2-22 are demographic and SES features, columns 23-2022 are LDA topic frequencies from tweets.
Labels: Columns 2123-2031 are health outcomes that you need to predict.

-----------------------------------------------------------------------

Demographic and SES features are:

['demo_pcblack', 'demo_pcfemale', 'demo_pchisp', 'demo_pcwht', 'demo_under18', 'demo_65over', 'largemetro', 'mediummetro', 'metrononmetro', 'micro', 'nchs_2013', 'ses_edu_coll', 'ses_foodenvt', 'ses_incomeratio', 'ses_log_hhinc', 'ses_pcaccess', 'ses_pcexerciss', 'ses_pchousing', 'ses_pcrural', 'ses_pcunemp', 'smallmetro']


LDA topic frequencies:
These columns contain frequency of the topic in that county. First row contains the topic ID, and each topic can be better understood by looking at the top 20 frequent words of that topic in the topics.csv file. 


Health Outcomes  are:

['health_aamort', 'health_fairpoor', 'health_mentunh', 'health_pcdiab', 'health_pcexcdrin', 'health_pcinact', 'health_pcsmoker', 'health_physunh', 'heath_pcobese']


training_data.mat

This file contains the same data as training_data.csv, but split into two matlab matrices named training_inputs and training_labels. Columns in matrices have the same order as in the csv file.

topics.csv

This file contains top 20 words for each topic. The topic IDs are in the first column and they correspond to the LDA topic frequencies columns in the training_data.csv file.
Each row is one topic, with topic ID in first column followed by top 20 most frequent words and their frequencies.
