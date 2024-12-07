# Version: 2024-12-07
# Author: Aliakbar Akbaritabar

# The goal here is to 
# show how precision and recall works to compare labels assigned by two methods.
# one is considered as "true" label and the other as "predicted".

import pandas as pd
# for precision and recall measurements
from sklearn import metrics as mt


# toy example data

row_number = [1, 2, 3, 4, 5, 6, 7, 8, 9]

names = ['Ali', 'Emilio', 'Tom', 'Julie', 'Ignacio', 'Ugo', 'Yuqi', 'Meagan', 'Andrea']

reported_gender = ['male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'male']

predicted_gender = ['female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female']

df = pd.DataFrame({'row_number': row_number, 
                   'names':names,
                    'reported_gender': reported_gender,
                    'predicted_gender': predicted_gender})

df

# how does precision and recall work?
# we calculate true positives, true negatives, false positives, and false negatives and use them in the formula for precision, recall, and f1-score that combines these two measures.

# let's do the calculation first for only males
# true positives for males, those reported male who are also predicted male. 
# Emilio, Tom, Ugo
TP_m = 1 + 1 + 1

# false positives for males, those reported female who are predicted male
# Yuqi
FP_m = 1

# false negatives for males, those reported male who are predicted female.
# Ali, Ignacio, Andrea
FN_m = 1 + 1 + 1

# true negatives for males, those reported female who are predicted female.
# Julie, Meagan
TN_m = 1 + 1

### now calculation for females
# true positives for females, those reported female who are also predicted female. 
# Julie, Meagan
TP_f = 1 + 1

# false positives for females, those reported male who are predicted female
# Ali, Ignacio, Andrea
FP_f = 1 + 1 + 1

# false negatives for females, those reported female who are predicted male.
# Yuqi
FN_f = 1

# true negatives for females, those reported male who are predicted male.
# Emilio, Tom, Ugo
TN_f = 1 + 1 + 1

# Precision, recall and f1-score for males
# TP_m / (TP_m + FP_m)
precision_male = TP_m / (TP_m + FP_m)
print(precision_male)

# TP_m / (TP_m + FN_m)
recall_male = TP_m / (TP_m + FN_m)
print(recall_male)

# f1_m = 2 * TP_m / (2 * TP_m + FP_m + FN_m)
f1_male = 2 * TP_m / (2 * TP_m + FP_m + FN_m)
print(f1_male)

# Precision, recall and f1-score for females
# TP_f / (TP_f + FP_f)
precision_female = TP_f / (TP_f + FP_f)
print(precision_female)

# TP_f / (TP_f + FN_f)
recall_female = TP_f / (TP_f + FN_f)
print(recall_female)

# f1_f = 2 * TP_f / (2 * TP_f + FP_f + FN_f)
f1_female = 2 * TP_f / (2 * TP_f + FP_f + FN_f)
print(f1_female)

# precision and recall without considering country
overall_prere = pd.DataFrame(mt.classification_report(df.reported_gender, df.predicted_gender, output_dict=True))

# add country name
overall_prere['countrycode'] = 'all_countries'

print(overall_prere)
# 	        female	    male	accuracy	macro avg	weighted_avg countrycode
# precision	0.400000	0.75	0.555556	0.575000	0.633333	all_countries
# recall	0.666667	0.50	0.555556	0.583333	0.555556	all_countries
# f1-score	0.500000	0.60	0.555556	0.550000	0.566667	all_countries
# support	3.000000	6.00	0.555556	9.000000	9.000000	all_countries


# Now let's add reported and affiliation countries
reported_country = ['Iran', 'Italy', 'Germany', 'Korea', 'Chile', 'Italy', 'China', 'USA', 'Italy']

affiliation_country = ['Germany', 'Germany', 'Germany', 'USA', 'UK', 'Germany', 'UK', 'USA', 'USA']

df_w_country = pd.DataFrame({'row_number': row_number, 
                   'names':names,
                    'reported_gender': reported_gender,
                    'predicted_gender': predicted_gender,
                    'reported_country': reported_country,
                    'affiliation_country': affiliation_country})

df_w_country


# precision and recall with considering country
# for each country, calculate precision and recall of gender classification
results_list = []

# grouped version of data
grouped_df = df_w_country.groupby('affiliation_country')

# with for loop of all countries, it took 77 minutes
for countrycode, country_dt in grouped_df:
    print(countrycode, '\n')
    res = pd.DataFrame(mt.classification_report(country_dt.reported_gender, country_dt.predicted_gender, output_dict=True))
    res['countrycode'] = countrycode
    results_list.append(res)

results_all = pd.concat(results_list)

# add overall countries
results_all = pd.concat([results_all, overall_prere])

# drop index and rename it
results_all = results_all.reset_index().rename(columns={'index':'metric'})

print(results_all)


## Visualize
import plotnine as gg

(
    gg.ggplot((results_all[results_all.metric == 'f1-score']),
    gg.aes('female', 'male')
    ) +
    gg.geom_point(gg.aes(size='weighted avg', color='factor(countrycode)'), alpha=0.4) +
    gg.geom_smooth(color='lightgreen', method='lm') +
    gg.geom_text(gg.aes(label='countrycode'), size=8) +
    gg.scale_x_continuous(limits=[0,1], labels=[0, .25, 0.5, .75, 1]) +
    gg.scale_y_continuous(limits=[0,1], labels=[0, .25, 0.5, .75, 1]) +
    gg.theme_classic() +
    gg.labs(x="F1 score of females", y="F1 score of males", title='Gender reported (True) versus Predicted', size='Weighted avg') +
    gg.theme_classic() +
    gg.theme(panel_background=gg.element_rect(fill='gray', alpha=.1), legend_position='none',
             axis_text_x=gg.element_text(size=8),
             axis_text_y=gg.element_text(hjust=1, size=10),
             axis_title_x=gg.element_text(size=10),
             axis_title_y=gg.element_text(size=10),
             strip_text_x=gg.element_text(size=10),
             figure_size=(6, 6))
)
